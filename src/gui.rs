/*! File containing logic for rendering immediate mode GUI */

use std::num::NonZeroU32;

use ash::{version::{EntryV1_0, DeviceV1_0, InstanceV1_0}, vk};
use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use crate::vulkan;
use crate::renderer;

pub struct Gui {
    pub context: Context,
    pub platform: WinitPlatform,
}

pub struct GuiRenderer {
    vk_pipeline: vk::Pipeline,
    vk_pipeline_layout: vk::PipelineLayout,
    
}

pub fn create_gui(window: &winit::window::Window) -> Gui {
    let mut context = Context::create();
    let mut platform = WinitPlatform::init(&mut context);
    platform.attach_window(context.io_mut(), window, HiDpiMode::Default);

    Gui {
        context,
        platform
    }
}

pub fn create_renderer(device: &ash::Device, 
    pool: vk::CommandPool, queue: vk::Queue,
    allocator: &vk_mem::Allocator, 
    gui: &Gui) -> GuiRenderer {
    let atlas = gui.context.fonts().build_rgba32_texture();

    let width = atlas.width;
    let height = atlas.height;
    let size = atlas.data.len();

    let font_image = vulkan::create_image(
        allocator, 
        atlas.width, atlas.height, 1, 1, 1u32.into(),
        vk::Format::R8G8B8A8_UNORM, vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::SharingMode::EXCLUSIVE, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    );

    let font_image_view = vulkan::create_image_view(
        device, &font_image, vk::ImageViewType::TYPE_2D,
        vk::Format::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR,
        1, 0, 1, 0
    );

    let staging_buffer = vulkan::create_buffer(
        allocator, size as u64,
        vk::SharingMode::EXCLUSIVE,
        vk::BufferUsageFlags::TRANSFER_SRC, 
        vk::MemoryPropertyFlags::HOST_VISIBLE | 
        vk::MemoryPropertyFlags::HOST_COHERENT
    );

    let raw_memory = allocator.map_memory(&staging_buffer.allocation).expect("Failed to map buffer data!");
    unsafe{ raw_memory.copy_from(atlas.data.as_ptr(), atlas.data.len()); }
    allocator.unmap_memory(&staging_buffer.allocation).expect("Failed to unmap buffer!");

    //create one-time command buffer
    let cmdbuf = vulkan::create_command_buffers(device, pool, vk::CommandBufferLevel::PRIMARY, 1);
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    //record command buffer
    unsafe { 
        device.begin_command_buffer(cmdbuf[0], &begin_info); 

        let font_image = vulkan::set_image_layout(device,
            cmdbuf[0], 
            font_image, 
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags::HOST, 
            vk::PipelineStageFlags::TRANSFER);

        vulkan::copy_buffer_to_image(device, cmdbuf[0], &font_image, vk::ImageAspectFlags::COLOR, &staging_buffer);

        let font_image = vulkan::set_image_layout(device, 
            cmdbuf[0], 
            font_image, 
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, 
            vk::ImageAspectFlags::COLOR, 
            vk::PipelineStageFlags::TRANSFER, 
            vk::PipelineStageFlags::FRAGMENT_SHADER);

        device.end_command_buffer(cmdbuf[0]);
    }

    //submit command buffer
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&cmdbuf)
        .build();
    let submits = [submit_info];
    unsafe { 
        device.queue_submit(queue, &submits, vk::Fence::null());
        device.queue_wait_idle(queue);
        device.free_command_buffers(pool, &cmdbuf);
        drop(cmdbuf);
    }
    
    //Create Font Texture Sampler

    //Create Descriptor Pool

    //Create Descriptor Set Layout

    //Create Descriptor Set

    //Create Pipeline
    //pipeline cache
    //pipeline layout
    //pipeline input assembly
    //pipeline rasterizer state
    //pipeline blend attachment
    //pipeline blend state
    //pipeline depth stencil
    //pipeline viewport state
    //pipeline dynamics
    //pipeline multisample
    //pipeline shader stages
    //pipeline vertex input description
    //pipeline renderpass
    //pipeline build

    todo!()
}

pub fn record_draw_commands(commandbuffer: vk::CommandBuffer, gui_data: &imgui::DrawData) {
    
}
