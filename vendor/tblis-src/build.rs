use std::path::PathBuf;

fn build_tblis() {
    println!("cargo:rerun-if-env-changed=TBLIS_DIR");
    println!("cargo:rerun-if-env-changed=TBLIS_SRC");
    println!("cargo:rerun-if-env-changed=TBLIS_VER");

    if !cfg!(feature = "build_from_source") || std::env::var_os("TBLIS_DIR").is_some() {
        return;
    }

    let tblis_src = std::env::var("TBLIS_SRC")
        .unwrap_or_else(|_| "https://github.com/MatthewsResearchGroup/tblis.git".to_owned());
    let tblis_ver = std::env::var("TBLIS_VER").unwrap_or_else(|_| "master".to_owned());
    let dst = cmake::Config::new("external_deps")
        .define("TBLIS_SRC", tblis_src)
        .define("TBLIS_VER", tblis_ver)
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
}

fn generate_link_search_paths(paths: &str) -> Vec<String> {
    let split_char = if cfg!(windows) { ";" } else { ":" };
    paths.split(split_char).map(str::to_owned).collect()
}

fn root_candidates(env_candidates: &[&str]) -> Vec<PathBuf> {
    let roots = ["/usr", "/usr/local", "/usr/local/share", "/opt"];
    env_candidates
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .flat_map(|path| generate_link_search_paths(&path))
        .filter(|path| !path.is_empty())
        .chain(roots.into_iter().map(str::to_owned))
        .map(PathBuf::from)
        .collect()
}

fn lib_candidates() -> impl Iterator<Item = PathBuf> {
    [
        "",
        "lib",
        "lib/stubs",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .into_iter()
    .map(PathBuf::from)
}

fn path_candidates(env_candidates: &[&str]) -> impl Iterator<Item = PathBuf> {
    root_candidates(env_candidates)
        .into_iter()
        .flat_map(|root| lib_candidates().map(move |lib| root.join(lib)))
        .filter(|path| path.exists())
        .filter_map(|path| std::fs::canonicalize(path).ok())
}

fn link_tblis() {
    let env_candidates = [
        "TBLIS_DIR",
        "REST_EXT_DIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "PATH",
    ];
    for path in path_candidates(&env_candidates) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static=tblis");
    } else {
        println!("cargo:rustc-link-lib=tblis");
    }
}

fn main() {
    build_tblis();
    link_tblis();
}
