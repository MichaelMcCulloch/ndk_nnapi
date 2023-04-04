
source .vscode/cargoNdkEnv.sh

architectures=(
    "x86_64-linux-android"
    "aarch64-linux-android"
)
#   "i686-linux-android:$JNI_LIB_DIR/x86"
#   "armv7-linux-androideabi:$JNI_LIB_DIR/armeabi-v7a"

build_types=(
    "--release"
    ""
)

for arch in "${architectures[@]}"; do
    
    
    for build_type in "${build_types[@]}"; do
        cargo ndk -p 30 --bindgen --target="$arch" build $build_type
    done
done