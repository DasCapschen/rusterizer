/*! File containing functions for general rendering code
 *  WILL directly interface with Vulkan, nothing else will be supported
 */
 
use std::cell::RefCell;

use crate::{gui, vulkan};
use ash::{version::{EntryV1_0, DeviceV1_0, InstanceV1_0}, vk};
use winit::window::Window;
use imgui_rs_vulkan_renderer::RendererVkContext;

struct Vk {
    entry: ash::Entry,
    instance: ash::Instance,

    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_families: vulkan::QueueFamilyIndices,

    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,

    swapchain_config: vulkan::SwapchainConfig,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,

    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,

    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

//needed for imgui renderer
impl RendererVkContext for Vk {
    fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    fn device(&self) -> &ash::Device {
        &self.device
    }

    fn queue(&self) -> vk::Queue {
        vulkan::get_graphics_queue(&self.device, &self.queue_families)
    }

    fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

//clean up vulkan resources
impl Drop for Vk {
    fn drop(&mut self) {
        //destroy things in opposite order of creation!
        unsafe {
            //destroy pipeline

            //destroy pipeline layout

            //destroy command pool
            self.device.destroy_command_pool(self.command_pool, None);

            //destroy framebuffers
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            //destroy render pass
            self.device.destroy_render_pass(self.render_pass, None);

            //destroy synchronisation
            self.device.destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_semaphore(self.render_finished_semaphore, None);

            //delete swapchain image views
            for img_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*img_view, None);
            }

            //destroy stuff created by device (fences, images, swapchain, ...)
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            //destroy device
            self.device.destroy_device(None);

            //destroy surface
            self.surface_loader.destroy_surface(self.surface, None);

            //destroy instance
            self.instance.destroy_instance(None);
        }
    }
}

//will hold Vulkan Device, Instance, etc
//is RefCell really the best idea, or would it be better if Renderer itself was just mutable?
pub struct Renderer {
    vk: Vk,
    imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe{ self.vk.device.device_wait_idle().expect("Cannot wait on device Idle!"); }

        self.imgui_renderer.destroy(&self.vk).expect("Cannot destroy imgui renderer!");
        //vk is automatically dropped hereafter
    }
}

pub fn create_renderer(window: &Window, gui_context: &mut imgui::Context) -> Renderer {
    //vulkan loader
    let entry = ash::Entry::new().expect("Failed to load Vulkan!");

    //vulkan
    let instance = vulkan::create_instance(&entry, window);

    //window surface
    let (surface_ext, surface) = vulkan::create_surface(&entry, &instance, &window);

    //vulkan device (gpu)
    let pdevice = vulkan::select_physical_device(&instance, &surface_ext, surface);
    let q_families = vulkan::find_queue_family_indices(&instance, pdevice, &surface_ext, surface);
    let ldevice = vulkan::create_logical_device(&instance, pdevice, &q_families);

    let size: (u32,u32) = window.inner_size().into();

    //swapchain
    let swapchain_ext = ash::extensions::khr::Swapchain::new(&instance, &ldevice);
    let swapchain_config = vulkan::get_swapchain_config(pdevice, &surface_ext, surface, size);
    let swapchain = vulkan::create_swapchain(&swapchain_ext, &swapchain_config, surface, &q_families);
    let (swapchain_images, swapchain_image_views) = vulkan::get_swapchain_images(&ldevice, &swapchain_ext, swapchain, &swapchain_config);

    //sync
    let image_available_semaphore = vulkan::create_semaphore(&ldevice);
    let render_finished_semaphore = vulkan::create_semaphore(&ldevice);

    //renderpass (TODO: rework this? is it even possible to have just 1 per application???)
    let render_pass = vulkan::create_render_pass(&ldevice, &swapchain_config);
    let framebuffers = swapchain_image_views.iter().map(|img_view| {
        vulkan::create_framebuffer(&ldevice, render_pass, *img_view, &swapchain_config) 
    }).collect::<Vec<_>>();

    //command buffers
    let command_pool = vulkan::create_command_pool(&ldevice, q_families.graphics.unwrap());
    let command_buffers = vulkan::create_command_buffers(&ldevice, command_pool, vk::CommandBufferLevel::PRIMARY, framebuffers.len() as u32);


    let vk = Vk {
        entry: entry,
        instance: instance,

        physical_device: pdevice,
        device: ldevice,

        queue_families: q_families,

        surface_loader: surface_ext,
        surface: surface,

        swapchain: swapchain,
        swapchain_loader: swapchain_ext,
        swapchain_config: swapchain_config,
        swapchain_images: swapchain_images,
        swapchain_image_views: swapchain_image_views,

        image_available_semaphore: image_available_semaphore,
        render_finished_semaphore: render_finished_semaphore,

        render_pass: render_pass,
        framebuffers: framebuffers,
        command_pool: command_pool,
        command_buffers: command_buffers
    };

    let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::new(&vk, 1, render_pass, gui_context).expect("Cannot create imgui renderer");

    Renderer {
        vk,
        imgui_renderer: /*RefCell::new(*/imgui_renderer//)
    }

}

pub fn draw_frame(renderer: &mut Renderer, gui_data: &imgui::DrawData) {
    //TODO: wait for Fence instead!
    unsafe{ renderer.vk.device.device_wait_idle().expect("Cannot wait for device idle!!"); }

    record_commands(renderer, gui_data);

    let result = unsafe {
        renderer.vk.swapchain_loader.acquire_next_image(
            renderer.vk.swapchain, 
            std::u64::MAX, 
            renderer.vk.image_available_semaphore, 
            vk::Fence::null())
    };

    if let Err(err_code) = result {
        if err_code == vk::Result::ERROR_OUT_OF_DATE_KHR {
            todo!("Recreate Swapchain!");
            return;
        } else {
            panic!("Cannot acquire swapchain image!!");
        }
    }

    let (image, suboptimal) = result.unwrap();
    if suboptimal {
        println!("Swapchain is suboptimal!");
    }

    submit_to_gpu(renderer, image);
    present_on_screen(renderer, image);
}

fn record_commands(renderer: &mut Renderer, gui_data: &imgui::DrawData) {
    //must do borrows before closure, because "closure requires "
    let imgui_renderer = &mut renderer.imgui_renderer;
    let vk = &renderer.vk;

    //should we enumerate instead of zipping and access with index? just use a for loop?
    renderer.vk.command_buffers.iter().zip(renderer.vk.framebuffers.iter()).for_each(|(cmdbuf, framebuf)|{

        //don't need any flags or inheritance yet
        let cmd_info = vk::CommandBufferBeginInfo::builder(); 

        unsafe { vk.device.begin_command_buffer(*cmdbuf, &cmd_info).expect("Cannot begin command buffer") }

        let render_area = vk::Rect2D::builder()
            .extent(vk.swapchain_config.extent)
            .offset(vk::Offset2D{ x: 0, y: 0 })
            .build();

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0]
            }
        }];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .framebuffer(*framebuf)
            .render_pass(vk.render_pass)
            .render_area(render_area)
            .clear_values(&clear_values);

        unsafe { vk.device.cmd_begin_render_pass(*cmdbuf, &render_pass_info, vk::SubpassContents::INLINE) }

        //TODO: draw something
        imgui_renderer.cmd_draw(vk, *cmdbuf, gui_data).expect("Cannot record imgui draw commands!");

        unsafe{ vk.device.cmd_end_render_pass(*cmdbuf) }
        unsafe{ vk.device.end_command_buffer(*cmdbuf).expect("Cannot end command buffer!") }
    });
}

fn submit_to_gpu(renderer: &Renderer, image: u32) {
    //vkQueueSubmit means "queue up these command buffers (submit these command buffers to this queue)"
    //it does NOT mean "submit this queue to the gpu"!

    let command_buffers = [renderer.vk.command_buffers[image as usize]];
    let wait_semaphores = [renderer.vk.image_available_semaphore];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let signal_semaphores = [renderer.vk.render_finished_semaphore];

    let submits = [
        vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .build()
    ];

    let graphics_queue = vulkan::get_graphics_queue(&renderer.vk.device, &renderer.vk.queue_families);
    unsafe{ renderer.vk.device.queue_submit(graphics_queue, &submits, vk::Fence::null()).expect("Cannot submit work to queue!") };
}

fn present_on_screen(renderer: &Renderer, image: u32) {
    //vkQueuePresent means "queue up this image for presentation on the screen"
    //does NOT mean "present this queue"

    let wait_semaphores = [renderer.vk.render_finished_semaphore];
    let swapchains = [renderer.vk.swapchain];
    let images = [image];

    let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&images);

    let present_queue = vulkan::get_present_queue(&renderer.vk.device, &renderer.vk.queue_families);
    let result = unsafe{renderer.vk.swapchain_loader.queue_present(present_queue, &present_info)};
    
    match result {
        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => { todo!("Recreate Swapchain!") },
        Err(_) => { panic!("Cannot queue up presentation!"); },
        _ => () //Ok(false) => everything is fine
    }
}

/* TODO:
    - setup render pass
    - setup subpasses (vkCmdNextSubpass())
        + one for drawing the scene
        + (possible post processing pass)
        + one for drawing the gui (depends on scene being done -> on top)

    - use primary command buffers for bigger rendering changes
        + switching renderpasses etc.
    - use secondary command buffers for actual drawing (record on separate threads!)
    - replay secondary cmdbufs into primary buf (vkCmdExecuteCommands())
    - submit primary cmdbuf

    => possibly leave secondary cmdbufs alone, only re-record when necessary
    => probably need to constantly re-record primary cmdbuf

    OR: possibly render UI into offscreen texture, then draw on top of scene ?

    - can switch pipelines quickly, so do it
    - just create a pipeline for every shader you have
    - use push constants / UBOs for shader "parameters" (textures, colors, etc)
    - sort objects in scene by pipeline -> less pieline switches, just binding new UBOs etc?
    - beginCommandBuffer, beginRenderpass, bindPipeline, renderObject, bindPipeline, renderObject, ... endRenderpass, endCommandBuffer

    IMGUI RENDERER:
    - begin command buffer
    - begin render pass
    - imgui-rs-vulkan-renderer::cmd_draw(context, command_buffer, draw_data)
    - end render pass
    - end command buffer
    - queue submit

    => records into same command buffer... hmmm :/
    Better:

    - record 2 secondary command buffers in parallel
        - scene command buffer
        - ui command buffer
    - wait for sync
    - begin comand buffer
    - begin render pass
    - executeCommands(scene)
    - next subpass
    - executeCommands(ui)
    - end render pass
    - end command buffer
    - queue submit

    Problem: imgui-rs-vulkan-renderer uses subpass 0 .-.
    or maybe we just give it a different renderpass? :thinking:
    or maybe it doesn't need to be in a different pass at all
    though I thought the commands can be reordered, so it might get drawn below the scene?


 */