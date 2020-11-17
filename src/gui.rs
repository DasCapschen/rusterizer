/*! File containing logic for rendering immediate mode GUI */

use ash::vk;
use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

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

pub fn record_draw_commands(commandbuffer: vk::CommandBuffer, gui_data: &imgui::DrawData) {
    
}
