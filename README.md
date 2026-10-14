# AENN

Personal Neural networking library for self-education. Implements Auto-Grad functionality, Tensors, Models, Training, and Inference.

## Third Party

googletest [https://github.com/google/googletest]</br>
imgui [https://github.com/ocornut/imgui/tree/docking]</br>
implot [http://github.com/epezent/implot]</br>
ImGuiFileDialog [https://github.com/aiekick/ImGuiFileDialog/tree/master]</br>

## How to Build

1. `git clone https://github.com/Kyojin-743/aenn`
2. Clone Thirdparty repos into thirdparty and add the custom CMakeLists.txt for imgui, implot, and ImGuiFileDialog
3. **DEPENDANCIES** This uses GLFW and OpenGL backend for ImGui, ensure GLFW and opengl are on your system:
   - (**Linux**): *Use appropriate Package Manager: `sudo apt-get update && sudo apt-get install libglfw3-dev`
   - (**Windows**):
      1. `git clone https://github.com/microsoft/vcpkg C:/vcpkg`
      2. `vcpkg install glfw3:x64-windows`
      3. Reload any terminals (Close vscode if opened) and run `$env:VCPKG_ROOT`, it should print out `C:\vcpkg`
4. `cd aenn ; cmake -B build`
5. `build_and_run.sh`(Linux) or `build_and_run.bat` (Windows)

## TODO

(In reverse order)

1. MNIST DEMO
2. Model library (to generate MLPs, CNNs, RNNs, etc..)
3. Auto-Grad library (wrapps tensors to facilitate training)
4. Tensor library
   - Along Axes (or whole tensor): These ops return new tensor
      - softmax, sigmoid

## Personal Notes

- Model is just the collection of Tensors (Auto-grad tensors, not the raw ones)
- optimizer shares ptr to the model's data (parameters)
- Training Loop:
  - predict (pass some input tensor through the model's layers, get output)
  - loss (take the predictions, and compute loss tensor)
  - backward (the loss is an auto-grad node that can have backward called on it(dumps the grad vaues to each tensor in the model))
  - step (apply, the gradiants to the model's tensors)
  - reset (zero-out the gradiant attached to the model's tensors)
