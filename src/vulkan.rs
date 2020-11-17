/*! File containing helper functions for Vulkan specific things */

use std::{collections::HashSet, ffi::{CStr, CString}};
use ash::{version::DeviceV1_0, vk::{self, QueueFamilyProperties}};
use ash::version::{EntryV1_0, InstanceV1_0};
use winit::window::Window;

pub struct QueueFamilyIndices {
    pub graphics: Option<u32>,
    pub present: Option<u32>,
    pub compute: Option<u32>,
    pub transfer: Option<u32>
}

pub struct SwapchainConfig {
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
    pub image_count: u32,
}

pub fn create_instance(entry: &ash::Entry, window: &Window) -> ash::Instance {

    let app_name = CString::new("DOWA-rs").unwrap();
    let app_version = vk::make_version(1, 0, 0);

    let app_info = vk::ApplicationInfo::builder()
        .api_version(vk::make_version(1,2,0))
        .application_name(&app_name)
        .application_version(app_version)
        .engine_name(&app_name)
        .engine_version(app_version);

    let extension_names = ash_window::enumerate_required_extensions(window)
        .unwrap().iter().map(|ext| ext.as_ptr()).collect::<Vec<_>>();

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);

    unsafe{ entry.create_instance(&instance_info, None).expect("Cannot create Vulkan instance!") }
}

pub fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> (ash::extensions::khr::Surface, vk::SurfaceKHR) {
    let surface = unsafe{ ash_window::create_surface(entry, instance, window, None).expect("Cannot create surface!") };
    let loader = ash::extensions::khr::Surface::new(entry, instance);
    (loader, surface)
}

pub fn find_queue_family_indices(instance: &ash::Instance, device: vk::PhysicalDevice, surface_ext: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR) -> QueueFamilyIndices {
    let fams = unsafe{ instance.get_physical_device_queue_family_properties(device) };

    //find queue family indices!
    let mut graphics_index: Option<u32> = None;
    let mut present_index: Option<u32> = None;
    for (i,fam) in fams.iter().enumerate() {
        if fam.queue_count > 0 && fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_index = Some(i as u32);
        }

        let present_support = unsafe{ 
            surface_ext.get_physical_device_surface_support(device, i as u32, surface)
                .expect("Cannot check device surface support!")
        };
        if fam.queue_count > 0 && present_support {
            present_index = Some(i as u32);
        }

        if present_index.is_some() && graphics_index.is_some() {
            break;
        }
    }

    QueueFamilyIndices {
        graphics: graphics_index,
        present: present_index,
        compute: None,
        transfer: None,
    }
}

pub fn select_physical_device(instance: &ash::Instance, surface_ext: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR) -> vk::PhysicalDevice {
    let devices = unsafe{ instance.enumerate_physical_devices().expect("Cannot enumerate devices!") };
    
    let found = devices.iter().find(|device| {

        //check if all extensions are supported
        let extensions = unsafe{ instance.enumerate_device_extension_properties(**device).expect("Cannot read device extensions!") };
        let extension_names = extensions.iter()
            .map(|ext| unsafe{ CStr::from_ptr(ext.extension_name.as_ptr()) } )
            .collect::<HashSet<&CStr>>();

        //FIXME: DO NOT HARDCODE!!!
        let mut requested_extension_names = HashSet::new();
        requested_extension_names.insert(ash::extensions::khr::Swapchain::name()); //need swapchain!

        let extensions_supported = requested_extension_names.is_subset(&extension_names);

        //check for certain features and properties that we wish to have!
        let props = unsafe{ instance.get_physical_device_properties(**device) };
        let feats = unsafe{ instance.get_physical_device_features(**device) };

        let features_supported = (feats.sampler_anisotropy != 0);

        //check for queue families
        let indices = find_queue_family_indices(instance, **device, surface_ext, surface);

        //select this device if it can both render graphics AND display them
        extensions_supported && features_supported && indices.graphics.is_some() && indices.present.is_some()
    });

    *found.expect("Found no GPU that supports vulkan!")
}

pub fn create_logical_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice, fams: &QueueFamilyIndices) -> ash::Device {
    let mut unique_fams = HashSet::new();
    unique_fams.insert(fams.graphics.unwrap());
    unique_fams.insert(fams.present.unwrap());
    //unique_fams.insert(fams.compute.unwrap());
    //unique_fams.insert(fams.transfer.unwrap());

    let mut queue_infos = vec![];
    let queue_priorities = [1.0f32]; //create 1 queue with priority 1.0
    for i in unique_fams {
        queue_infos.push(vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(i)
            .queue_priorities(&queue_priorities) 
            .build()); 
    }

    //enable features (CHECK IN SELECT_PHYSICAL_DEVICE FIRST!!!)
    let features = vk::PhysicalDeviceFeatures {
        sampler_anisotropy: vk::TRUE,
        .. Default::default()
    };

    //FIXME: DO NOT HARDCODE!!!
    let extension_names = [ash::extensions::khr::Swapchain::name().as_ptr()];

    let device_info = vk::DeviceCreateInfo::builder()
        .enabled_features(&features)
        .enabled_extension_names(&extension_names)
        .queue_create_infos(&queue_infos);

    unsafe{ instance.create_device(physical_device, &device_info, None).expect("Cannot create logical device!") }
}

pub fn get_graphics_queue(device: &ash::Device, fams: &QueueFamilyIndices) -> vk::Queue  {
    unsafe{ device.get_device_queue(fams.graphics.unwrap(), 0) }
}

pub fn get_present_queue(device: &ash::Device, fams: &QueueFamilyIndices) -> vk::Queue  {
    unsafe{ device.get_device_queue(fams.present.unwrap(), 0) }
}

pub fn get_swapchain_config(physical_device: vk::PhysicalDevice, 
    surface_ext: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR,
    (width, height): (u32, u32)) -> SwapchainConfig {

    let caps = unsafe{ surface_ext.get_physical_device_surface_capabilities(physical_device, surface).expect("Cannot get surface capabilities!") };
    
    let formats = unsafe{ surface_ext.get_physical_device_surface_formats(physical_device, surface).expect("Cannot get surface formats!") };
    if formats.is_empty() {
        panic!("Device supports no surface formats!");
    }
    
    let modes = unsafe{ surface_ext.get_physical_device_surface_present_modes(physical_device, surface).expect("Cannot get surface present modes!") };
    if modes.is_empty() {
        panic!("Device supports no surface present modes!");
    }

    //SELECT FORMAT
    let surface_format;
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        //surface has no preferred format, choose at will
        surface_format = vk::SurfaceFormatKHR::builder()
            .format(vk::Format::B8G8R8A8_UNORM)
            .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .build(); //should be safe, all constants!
    } else {
        surface_format = *formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8A8_UNORM 
            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }).unwrap_or(&formats[0]);
    }

    //SELECT PRESENT MODE
    let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX //would be best
    } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE //better than fifo
    } else { 
        vk::PresentModeKHR::FIFO //must exist
    };

    //SELECT EXTEND
    let extent = if caps.current_extent.width != std::u32::MAX {
        caps.current_extent
    } else {
        //width, height are ACTUAL size
        //clamp to max/min :)
        //hint: clamp() is unstable ... .-.
        let w = caps.min_image_extent.width.max(caps.max_image_extent.width.min(width));
        let h = caps.min_image_extent.height.max(caps.max_image_extent.height.min(height));
        vk::Extent2D::builder().width(w).height(h).build()
    };

    let image_count = if caps.max_image_count > 0 {
        (caps.min_image_count+1).min(caps.max_image_count)
    } else {
        caps.min_image_count+1
    };

    SwapchainConfig {
        surface_format,
        present_mode,
        extent,
        image_count
    }
}

pub fn create_swapchain(swapchain_ext: &ash::extensions::khr::Swapchain,
    config: &SwapchainConfig,
    surface: vk::SurfaceKHR,
    queue_family_indices: &QueueFamilyIndices
) -> vk::SwapchainKHR {

    let fam_indices;
    let sharing_mode;

    if queue_family_indices.graphics == queue_family_indices.present {
        fam_indices = vec![queue_family_indices.graphics.unwrap()];
        sharing_mode = vk::SharingMode::EXCLUSIVE;
    } else {
        fam_indices = vec![queue_family_indices.graphics.unwrap(), queue_family_indices.present.unwrap()];
        sharing_mode = vk::SharingMode::CONCURRENT;
    }

    let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(config.image_count)
        .image_format(config.surface_format.format)
        .image_color_space(config.surface_format.color_space)
        .image_extent(config.extent)
        .present_mode(config.present_mode)
        .image_array_layers(1)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(&fam_indices)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .clipped(true); //clip pixels hidden by other windows

    unsafe{ swapchain_ext.create_swapchain(&swapchain_info, None).expect("Cannot create swapchain!") }
}

pub fn get_swapchain_images(
    device: &ash::Device,
    swapchain_ext: &ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_config: &SwapchainConfig
) -> (Vec<vk::Image>, Vec<vk::ImageView>) {
    
    let images = unsafe{ swapchain_ext.get_swapchain_images(swapchain).expect("Cannot get swapchain images!") };

    let views = images.iter().map(|image| {
        let subres = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build(); //why? I thought they implement deref...

        let swizzle = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY)
            .build();

        let info = vk::ImageViewCreateInfo::builder()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_config.surface_format.format)
            .subresource_range(subres)
            .components(swizzle);

        unsafe{ device.create_image_view(&info, None).expect("Cannot create Swapchain Image Views!") }
    }).collect();

    (images, views)
}

pub fn create_semaphore(device: &ash::Device) -> vk::Semaphore {
    //there is no info to pass for semaphore creation (yet)
    let info = vk::SemaphoreCreateInfo::builder();
    unsafe{ device.create_semaphore(&info, None).expect("Cannot create semaphore!") }
}

pub fn create_fence(device: &ash::Device, signalled: bool) -> vk::Fence {
    let mut info = vk::FenceCreateInfo::builder();

    if signalled {
        info = info.flags(vk::FenceCreateFlags::SIGNALED);
    }

    unsafe { device.create_fence(&info, None).expect("Cannot create Fence!") }
}

/** Creates a renderpass with a single subpass and a single color attachment */
pub fn create_render_pass(device: &ash::Device, swapchain_config: &SwapchainConfig) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_config.surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build(); //should be fine, only constants in this struct

    let color_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build(); //should be fine, only constants in this struct

    //initial layout = before renderpass
    //reference layout = during renderpass
    //final layout = after renderpass
    
    let subpass0_attachements = [color_ref];
    let subpass0 = vk::SubpassDescription::builder()
        .color_attachments(&subpass0_attachements)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build();
    

    let attachments = [color_attachment];
    let subpasses = [subpass0];
    let renderpass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses);

    unsafe{ device.create_render_pass(&renderpass_info, None).expect("Cannot create Render Pass!") }
}

/** Creates a framebuffer referencing a single color attachment */
pub fn create_framebuffer(device: &ash::Device, render_pass: vk::RenderPass, image_view: vk::ImageView, swapchain_config: &SwapchainConfig) -> vk::Framebuffer {
    let attachments = [image_view];

    let create_info = vk::FramebufferCreateInfo::builder()
        .attachments(&attachments)
        .render_pass(render_pass)
        .width(swapchain_config.extent.width)
        .height(swapchain_config.extent.height)
        .layers(1);

    unsafe{ device.create_framebuffer(&create_info, None).expect("Cannot create Framebuffer!") }
}

pub fn create_command_pool(device: &ash::Device, queue_index: u32) -> vk::CommandPool {
    let info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER); //allows resetting a single cmdbuf

    unsafe{ device.create_command_pool(&info, None).expect("Cannot create command pool") }
}

pub fn create_command_buffers(device: &ash::Device, command_pool: vk::CommandPool, level: vk::CommandBufferLevel, count: u32) -> Vec<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .command_buffer_count(count)
        .level(level);
    
    unsafe{ device.allocate_command_buffers(&info).expect("Cannot allocate command buffers!") }
}