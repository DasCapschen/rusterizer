/*! File containing main engine loop */

use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use std::time::{Duration, Instant};

use crate::gui;
use crate::renderer;

#[derive(Copy, Clone)]
pub struct Arguments {
    pub width: u16,
    pub height: u16,
    pub fps: u16,
    pub msaa: u16,
}

pub struct GuiState {
    demo_open: bool,
}

fn construct_window(width: u16, height: u16) -> (EventLoop<()>, Window) {
    let event_loop = EventLoop::new();

    let size = LogicalSize::new(width, height);

    let window: Window = WindowBuilder::new()
        .with_title("DOWA Rust")
        .with_resizable(false)
        .with_inner_size(size)
        .build(&event_loop)
        .unwrap();

    (event_loop, window)
}

pub fn start_engine(arguments: Arguments) {
    let (event_loop, window) = construct_window(arguments.width, arguments.height);

    //TODO: i really don't like that we are drawing GUI *here*, that, and state, should be elsewhere!!
    let mut gui_state = GuiState { demo_open: true };
    let mut gui = gui::create_gui(&window);
    //TODO: imgui setup (fonts, themes etc.)
    let mut renderer = renderer::create_renderer(&window, &mut gui.context);

    let one_over_sixty = Duration::from_secs_f64(1.0 / 60.0);
    let frame_time_limit = Duration::from_secs_f64(1.0 / arguments.fps as f64);

    let mut fixed_instant = Instant::now();
    let mut render_instant = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit
            },
            Event::MainEventsCleared => {
                //do any functions that were queued last frame (like delete)
                //how would this work?? maybe just don't have it
                //work_function_queue();

                //process input etc
                handle_events();

                if fixed_instant.elapsed() >= one_over_sixty {
                    fixed_update_scene(one_over_sixty.as_secs_f64());
                }

                //actually render the image, but dont exceed fps limit
                if render_instant.elapsed() >= frame_time_limit {
                    let delta_time = render_instant.elapsed().as_secs_f64();
                    render_instant = Instant::now();

                    //update scene
                    update_scene(delta_time);

                    //update gui
                    let gui_frame = update_gui(&mut gui.context, &mut gui.platform, &window, delta_time, &mut gui_state);
                    let gui_data = gui_frame.render();

                    //render to screen
                    renderer::draw_frame(&mut renderer, gui_data)
                }
            },
            Event::RedrawRequested(_) => { /* redraw, either when window.request_redraw(), or e.g. resized */ },
            Event::LoopDestroyed => { /* DO ANY CLEANUP HERE BEFORE APPLICATION EXIT */}
            //in any other case...
            event => {
                //let gui handle the event (first? really?)
                gui.platform.handle_event(gui.context.io_mut(), &window, &event);

                //if input event, handle it
                if let Event::DeviceEvent{event: ev, device_id: id} = event {
                    raw_input(ev, id);
                }
            }
        }
    });
}

fn raw_input(event: DeviceEvent, id: DeviceId) {
    match event {
        //Added, Removed -> new device registered
        //Motion -> any motion (mouse, analog stick, wheel)
        //Button -> any button (controller, keyboard)
        DeviceEvent::MouseMotion { delta } => {}
        DeviceEvent::MouseWheel { delta } => {}
        DeviceEvent::Key(key) => {}
        _ => (),
    }
}

fn handle_events() {
    // for event in events
    //     for action in actions[event.id]
    //         action.execute()
    //
    // action -> some trait object for input events
    //        -> created by game code
    // (or just storing a function pointer??)
    // maybe pass a closure to action's constructor???
}

fn fixed_update_scene(delta_time: f64) {
    // for object in fixed_update_objects
    //     object.fixed_update(delta_time)

    // game code does it's thing
    // or maybe implement an ECS instead?
}

fn update_scene(delta_time: f64) {
    // for object in update_objects
    //     object.update(delta_time)

    // game code does it's thing
    // or maybe implement an ECS instead?
}

// 'a lifetime -> return value only lives as long as context lives.
// as soon as context (in caller) goes out of scope, return value cannot be used anymore!
fn update_gui<'a>(
    context: &'a mut imgui::Context,
    platform: &mut imgui_winit_support::WinitPlatform,
    window: &Window,
    delta_time: f64,
    gui_state: &mut GuiState,
) -> imgui::Ui<'a> {
    context
        .io_mut()
        .update_delta_time(Duration::from_secs_f64(delta_time));
    platform
        .prepare_frame(context.io_mut(), &window)
        .expect("Couldn't prepare GUI frame!");
    let ui_frame = context.frame();

    //DRAW UI
    ui_frame.show_demo_window(&mut gui_state.demo_open);

    platform.prepare_render(&ui_frame, &window);
    ui_frame
}

// use struct of arrays ECS ?
// Entity == index
// have arrays of components
// what if an entity doesn't have some component? the indices wouldn't be right anymore
// use a HashMap ? Entity = Key ?
