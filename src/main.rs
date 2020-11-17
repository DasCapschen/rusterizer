/*! File containing application startup */

extern crate winit;
extern crate ash;
extern crate ash_window;
extern crate vk_mem;
extern crate vk_sync;
extern crate imgui;
extern crate imgui_winit_support;
extern crate imgui_rs_vulkan_renderer;

mod engine;
mod renderer;
mod vulkan;
mod gui;

//this should probably go somewhere else to be available globally
enum ErrorCodes {
    Success = 0,
    Usage = 1,
}

fn usage_err(message: &str) -> ! {
    println!("Error: {}", message);
    println!();
    usage();
}

fn usage() -> ! {
    println!("Usage: ./DOWAEngine [OPTIONS]");
    println!("Available Options:");
    println!("    -w, --width NUMBER        set window width");
    println!("    -h, --height NUMBER       set window height");
    println!("    --fps NUMBER              set fps limit");
    println!("    --msaa NUMBER             request MSAA samples");
    println!("    --help                    show this help");
    std::process::exit(ErrorCodes::Usage as i32);
}

fn parse_u16(opt: &Option<String>) -> u16 {
    if let Some(string) = opt {
        if let Ok(int) = string.parse::<u16>() {
            int
        }
        else {
            usage_err("Argument was not a number!");
        }
    }
    else {
        usage_err("Argument missing!");
    }
}

fn main() {
    
    let mut args = std::env::args();

    let mut arguments = engine::Arguments {
        width: 1280,
        height: 720,
        fps: 144,
        msaa: 1,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-w" | "--width" => {
                arguments.width = parse_u16(&args.next());
            },
            "-h" | "--height" => {  
                arguments.height = parse_u16(&args.next());
            },
            "--fps" => {
                arguments.fps = parse_u16(&args.next());
            },
            "--msaa" => {
                arguments.msaa = parse_u16(&args.next());
            },
            "--help" => {
                usage();
            },
            _ => {},
        }
    }

    //start main game loop
    engine::start_engine(arguments);
}