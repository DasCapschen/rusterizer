/*! File containing logic for rendering immediate mode GUI */

use std::{mem::size_of, num::NonZeroU32};

use crate::renderer;
use crate::vulkan;
use ash::{
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

pub struct Gui {
    pub context: Context,
    pub platform: WinitPlatform,
}

pub struct GuiRenderer {
    pipeline: vulkan::Pipeline,
    font_image: vulkan::AllocatedImage,
    font_image_view: vk::ImageView,
    font_sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    vertex_buffer: Option<vulkan::AllocatedBuffer>,
    index_buffer: Option<vulkan::AllocatedBuffer>,
    vertex_count: i32,
    index_count: i32,
    push_constants: ImguiPushConstants,
}

struct ImguiPushConstants {
    scale: imgui::sys::ImVec2,
    translate: imgui::sys::ImVec2,
}

pub fn create_gui(window: &winit::window::Window) -> Gui {
    let mut context = Context::create();
    let mut platform = WinitPlatform::init(&mut context);
    platform.attach_window(context.io_mut(), window, HiDpiMode::Default);

    Gui { context, platform }
}

pub fn create_renderer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    render_pass: vk::RenderPass,
    subpass: u32,
    allocator: &vk_mem::Allocator,
    gui: &mut Gui,
) -> GuiRenderer {
    let width = gui.context.io().display_size[0];
    let height = gui.context.io().display_size[1];

    let mut fonts = gui.context.fonts();
    let atlas = fonts.build_rgba32_texture();

    let mut font_image = vulkan::create_image(
        allocator,
        atlas.width,
        atlas.height,
        1,
        1,
        1u32.into(),
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::SharingMode::EXCLUSIVE,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    let font_image_view = vulkan::create_image_view(
        device,
        &font_image,
        vk::ImageViewType::TYPE_2D,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        1,
        0,
        1,
        0,
    );

    let staging_buffer = vulkan::create_buffer(
        allocator,
        atlas.data.len() as u64,
        vk::SharingMode::EXCLUSIVE,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    let raw_memory = allocator
        .map_memory(&staging_buffer.allocation)
        .expect("Failed to map buffer data!");
    unsafe {
        raw_memory.copy_from(atlas.data.as_ptr(), atlas.data.len());
    }
    allocator
        .unmap_memory(&staging_buffer.allocation)
        .expect("Failed to unmap buffer!");

    //create one-time command buffer
    let cmdbuf = vulkan::create_command_buffers(device, pool, vk::CommandBufferLevel::PRIMARY, 1);
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    //record command buffer
    unsafe {
        device.begin_command_buffer(cmdbuf[0], &begin_info);

        font_image = vulkan::set_image_layout(
            device,
            cmdbuf[0],
            font_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
        );

        vulkan::copy_buffer_to_image(
            device,
            cmdbuf[0],
            &font_image,
            vk::ImageAspectFlags::COLOR,
            &staging_buffer,
        );

        font_image = vulkan::set_image_layout(
            device,
            cmdbuf[0],
            font_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );

        device.end_command_buffer(cmdbuf[0]);
    }

    //submit command buffer
    let submit_info = vk::SubmitInfo::builder().command_buffers(&cmdbuf).build();
    let submits = [submit_info];
    unsafe {
        device.queue_submit(queue, &submits, vk::Fence::null());
        device.queue_wait_idle(queue);
        device.free_command_buffers(pool, &cmdbuf);
        drop(cmdbuf);
    }

    //Create Font Texture Sampler
    let font_sampler = vulkan::create_texture_sampler(
        device,
        vulkan::SamplerType::TRILINEAR,
        vk::SamplerAddressMode::CLAMP_TO_EDGE,
        0.0,
        vk::BorderColor::FLOAT_OPAQUE_WHITE,
    );

    //Create Descriptor Pool
    let descriptor_pool =
        vulkan::create_descriptor_pool(device, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, 2);

    //Create Descriptor Set Layout
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();

    let descriptor_set_layout = vulkan::create_descriptor_set_layout(device, &[binding]);

    //Create Descriptor Set
    let descriptor_set =
        vulkan::create_desciptor_sets(device, descriptor_pool, &[descriptor_set_layout]);

    //update font descriptor in set
    let font_descriptor = vk::DescriptorImageInfo::builder()
        .sampler(font_sampler)
        .image_view(font_image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();

    let image_infos = [font_descriptor];

    let write_descriptor = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set[0])
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)
        .build();

    let writes = [write_descriptor];
    let copies = [];
    unsafe {
        device.update_descriptor_sets(&writes, &copies);
    }

    let stencil_op = vk::StencilOpState::builder()
        .compare_op(vk::CompareOp::ALWAYS)
        .build();

    //Create Pipeline
    let pipeline = vulkan::Pipeline::builder(device)
        .cache()
        .add_push_constant_range(
            vk::ShaderStageFlags::VERTEX,
            0,
            size_of::<ImguiPushConstants>() as u32,
        )
        .add_descriptor_set_layout(descriptor_set_layout)
        .input_assembly(vk::PrimitiveTopology::TRIANGLE_LIST, false)
        .rasterizer(
            false,
            false,
            false,
            0.0,
            0.0,
            0.0,
            1.0,
            vk::PolygonMode::FILL,
            vk::FrontFace::COUNTER_CLOCKWISE,
            vk::CullModeFlags::NONE,
        )
        .add_color_blend_attachment(
            true,
            vk::BlendFactor::SRC_ALPHA,
            vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            vk::BlendOp::ADD,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            vk::BlendFactor::ZERO,
            vk::BlendOp::ADD,
            vk::ColorComponentFlags::all(),
        )
        .color_blend_info(false, vk::LogicOp::NO_OP, [0f32, 0f32, 0f32, 0f32])
        .depth_stencil(
            false,
            false,
            false,
            false,
            vk::CompareOp::LESS_OR_EQUAL,
            0f32,
            1f32,
            stencil_op,
            stencil_op,
        )
        .add_viewport(0.0, 0.0, width, height, 0.0, 1.0)
        .add_scissor(0, 0, width as u32, height as u32)
        .add_dynamic_state(vk::DynamicState::VIEWPORT)
        .add_dynamic_state(vk::DynamicState::SCISSOR)
        .multisample(vk::SampleCountFlags::TYPE_1, false, 0.0, false, false, None)
        .add_shader_stage("res/imgui.vert", vk::ShaderStageFlags::VERTEX)
        .add_shader_stage("res/imgui.frag", vk::ShaderStageFlags::FRAGMENT)
        .add_vertex_binding(
            0,
            size_of::<imgui::sys::ImDrawVert>() as u32,
            vk::VertexInputRate::VERTEX,
        )
        .add_vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT, 0)
        .add_vertex_attribute(
            1,
            0,
            vk::Format::R32G32_SFLOAT,
            size_of::<imgui::sys::ImVec2>() as u32,
        )
        .add_vertex_attribute(
            2,
            0,
            vk::Format::R8G8B8A8_UNORM,
            2 * size_of::<imgui::sys::ImVec2>() as u32,
        )
        .render_pass(render_pass, subpass)
        .build_graphics();

    GuiRenderer {
        pipeline,
        font_image,
        font_image_view,
        font_sampler,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_set: descriptor_set[0],
        vertex_buffer: None,
        index_buffer: None,
        vertex_count: 0,
        index_count: 0,
        push_constants: ImguiPushConstants {
            scale: imgui::sys::ImVec2::new(1.0, 1.0),
            translate: imgui::sys::ImVec2::new(-1.0, -1.0),
        },
    }
}

pub fn record_draw_commands(commandbuffer: vk::CommandBuffer, gui_data: &imgui::DrawData) {}
