use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=data/");

    let source = Path::new("data");
    if !source.exists() {
        return;
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let destination = target_dir.join("data");

    if destination.exists() {
        fs::remove_dir_all(&destination).unwrap();
    }

    fs::create_dir_all(&destination).unwrap();
    if let Ok(entries) = fs::read_dir(source) {
        for entry in entries {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                fs::copy(&path, destination.join(path.file_name().unwrap())).unwrap();
            }
        }
    }
}
