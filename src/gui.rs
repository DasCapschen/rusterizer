/*! File containing logic for rendering immediate mode GUI */

use std::{mem::size_of};


use crate::vulkan;
use ash::{
    version::{DeviceV1_0},
    vk,
};
use imgui::{Context, DrawIdx, DrawVert};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

pub struct Gui {
    pub context: Context,
    pub platform: WinitPlatform,
}

//TODO: somehow implement drop for everything!
//in theory we can just have everything have a 'a lifetime of the device
//and a reference to it
//since the device will always drop last.

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
impl GuiRenderer {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.font_image_view, None);
            device.destroy_sampler(self.font_sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }

        self.pipeline.destroy(device);

        self.font_image.destroy(allocator);

        if self.vertex_buffer.is_some() {
            self.vertex_buffer.as_mut().unwrap().destroy(allocator);
        }
        if self.index_buffer.is_some() {
            self.index_buffer.as_mut().unwrap().destroy(allocator);
        }
    }
}

struct ImguiPushConstants {
    scale: [f32;2],
    translate: [f32;2],
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

    let mut staging_buffer = vulkan::create_buffer(
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
    allocator.unmap_memory(&staging_buffer.allocation);

    //create one-time command buffer
    let cmdbuf = vulkan::create_command_buffers(device, pool, vk::CommandBufferLevel::PRIMARY, 1);
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    //record command buffer
    unsafe {
        device.begin_command_buffer(cmdbuf[0], &begin_info).expect("Cannot begin command buffer!");

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

        device.end_command_buffer(cmdbuf[0]).expect("Cannot end command buffer!");
    }

    //submit command buffer
    let submit_info = vk::SubmitInfo::builder().command_buffers(&cmdbuf).build();
    let submits = [submit_info];
    unsafe {
        device.queue_submit(queue, &submits, vk::Fence::null()).expect("Queue submit failed!");
        device.queue_wait_idle(queue).expect("Failed to wait for queue");
        device.free_command_buffers(pool, &cmdbuf);
        drop(cmdbuf);

        staging_buffer.destroy(allocator);
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
        .add_shader_stage("res/imgui.vert.spv", vk::ShaderStageFlags::VERTEX)
        .add_shader_stage("res/imgui.frag.spv", vk::ShaderStageFlags::FRAGMENT)
        .add_vertex_binding(
            0,
            size_of::<DrawVert>() as u32,
            vk::VertexInputRate::VERTEX,
        )
        .add_vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT, 0)
        .add_vertex_attribute(
            1,
            0,
            vk::Format::R32G32_SFLOAT,
            size_of::<[f32;2]>() as u32,
        )
        .add_vertex_attribute(
            2,
            0,
            vk::Format::R8G8B8A8_UNORM,
            2 * size_of::<[f32;2]>() as u32,
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
            scale: [1.0, 1.0],
            translate: [-1.0, -1.0],
        },
    }
}

pub fn record_draw_commands(
    renderer: &mut GuiRenderer, 
    allocator: &vk_mem::Allocator,
    device: &ash::Device,
    commandbuffer: vk::CommandBuffer, 
    draw_data: &imgui::DrawData
) {
    // 1) Update buffers
    // 2) Draw Frame
    
    // UPDATE BUFFERS
    let vcount = draw_data.total_vtx_count;
    let icount = draw_data.total_idx_count;
    let vbufsize = vcount as u64 * size_of::<DrawVert>() as u64;
    let ibufsize = icount as u64 * size_of::<DrawIdx>() as u64;

    if vbufsize == 0 || ibufsize == 0 {
        return;
    }

    if renderer.vertex_buffer.is_none() || renderer.vertex_count != vcount {
        if let Some(buf) = &mut renderer.vertex_buffer {
            buf.destroy(&allocator);
        }

        renderer.vertex_buffer = Some(vulkan::create_buffer(allocator, vbufsize, vk::SharingMode::EXCLUSIVE, vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE));
        renderer.vertex_count = vcount;
    }

    if renderer.index_buffer.is_none() || renderer.index_count != icount {
        if let Some(buf) = &mut renderer.index_buffer {
            buf.destroy(&allocator);
        }

        renderer.index_buffer = Some(vulkan::create_buffer(allocator, ibufsize, vk::SharingMode::EXCLUSIVE, vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE));
        renderer.index_count = icount;
    }

    let mut vertex_dst = allocator.map_memory(&renderer.vertex_buffer.as_ref().unwrap().allocation).expect("Failed to map vertex buffer!") as (*mut DrawVert);
    let mut index_dst = allocator.map_memory(&renderer.index_buffer.as_ref().unwrap().allocation).expect("Failed to map index buffer!") as (*mut DrawIdx);

    //god i hope this works as expected...
    draw_data.draw_lists().for_each(|drawlist| {
        unsafe {
            //copy a bit of data
            vertex_dst.copy_from(drawlist.vtx_buffer().as_ptr(), drawlist.vtx_buffer().len() as usize);
            index_dst.copy_from(drawlist.idx_buffer().as_ptr(), drawlist.idx_buffer().len() as usize);
            //step ahead so we don't overwrite next iteration
            vertex_dst = vertex_dst.offset(drawlist.vtx_buffer().len() as isize);
            index_dst = index_dst.offset(drawlist.idx_buffer().len() as isize);
        }
    });

    allocator.flush_allocation(&renderer.vertex_buffer.as_ref().unwrap().allocation, 0, vk::WHOLE_SIZE as usize);
    allocator.flush_allocation(&renderer.index_buffer.as_ref().unwrap().allocation, 0, vk::WHOLE_SIZE as usize);
    allocator.unmap_memory(&renderer.vertex_buffer.as_ref().unwrap().allocation);
    allocator.unmap_memory(&renderer.index_buffer.as_ref().unwrap().allocation);


    //DRAW FRAME
    let desc_sets = [renderer.descriptor_set];
    let dyn_offsets = [];

    let width = draw_data.framebuffer_scale[0] * draw_data.display_size[0];
    let height = draw_data.framebuffer_scale[1] * draw_data.display_size[1];

    let viewports = [
        vk::Viewport::builder()
        .width(width)
        .height(height)
        .x(0.0)
        .y(0.0)
        .min_depth(0.0)
        .max_depth(1.0)
        .build()
    ];

    renderer.push_constants.scale = [ 2.0 / width, 2.0 / height ];
    renderer.push_constants.translate = [ -1.0, -1.0 ];

    unsafe { 
        device.cmd_bind_descriptor_sets(
            commandbuffer, 
            renderer.pipeline.bind_point, 
            renderer.pipeline.layout, 
            0, &desc_sets,
            &dyn_offsets);

        device.cmd_bind_pipeline(
            commandbuffer, 
            renderer.pipeline.bind_point, 
            renderer.pipeline.pipeline);

        device.cmd_set_viewport(commandbuffer, 
            0, &viewports);

        //wow... really? seems unsafe xD
        let push_const_slice = std::slice::from_raw_parts(
            (&renderer.push_constants as *const ImguiPushConstants) as *const u8,
            size_of::<ImguiPushConstants>()
        );

        device.cmd_push_constants(commandbuffer, 
            renderer.pipeline.layout, 
            vk::ShaderStageFlags::VERTEX, 
            0, 
            push_const_slice
        );

        let vbuffers = [renderer.vertex_buffer.as_ref().unwrap().buffer];
        let offsets = [0];
        device.cmd_bind_vertex_buffers(
            commandbuffer, 
            0, 
            &vbuffers, 
            &offsets);
        device.cmd_bind_index_buffer(
            commandbuffer, 
            renderer.index_buffer.as_ref().unwrap().buffer, 
            0, 
            vk::IndexType::UINT16);
            //pub type DrawIdx = sys::ImDrawIdx;
            //pub type ImDrawIdx = ::std::os::raw::c_ushort;
            //pub type c_ushort = u16;
    }

    let mut vertex_offset = 0;
    let mut index_offset = 0;

    draw_data.draw_lists().for_each(|draw_list| {
        draw_list.commands().for_each(|command| {
            match command {
                imgui::DrawCmd::Elements {
                    count, cmd_params
                } => {

                    //is this correct? imgui-rs-vk-renderer does a lot more calculations...
                    let scissor_offset = vk::Offset2D::builder()
                        .x(cmd_params.clip_rect[0] as i32)
                        .y(cmd_params.clip_rect[1] as i32)
                        .build();

                    let scissor_extent = vk::Extent2D::builder()
                        .width((cmd_params.clip_rect[2] - cmd_params.clip_rect[0]) as u32)
                        .height((cmd_params.clip_rect[3] - cmd_params.clip_rect[1]) as u32)
                        .build();
        
                    let scissors = [vk::Rect2D::builder()
                        .offset(scissor_offset)
                        .extent(scissor_extent)
                        .build()];

                    unsafe {
                        device.cmd_set_scissor(
                          commandbuffer, 
                          0, &scissors
                        );
                        device.cmd_draw_indexed(
                            commandbuffer, 
                            count as u32, 
                            1, 
                            index_offset + cmd_params.idx_offset as u32, 
                            vertex_offset + cmd_params.vtx_offset as i32, 
                            0
                        );
                    }
                },
                _ => ()
            }
        });

        index_offset += draw_list.idx_buffer().len() as u32;
        vertex_offset += draw_list.vtx_buffer().len() as i32;
    });
    
}
