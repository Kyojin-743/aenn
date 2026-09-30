# AENN

Personal Neural networking library for self-education. Implements Auto-Grad functionality, Tensors, Models, Training, and Inference.

## Third Party

GTest [https://github.com/google/googletest]</br>
Tensor Library [https://github.com/abeschneider/tensor]</br>

## TODO

(In reverse order)

1. MNIST DEMO
2. Model library (to generate MLPs, CNNs, RNNs, etc..)
3. Auto-Grad library (wrapps tensors to facilitate training)
4. Tensor library
   - Along Axes (or whole tensor): These ops return new tensor
      - Sum, Mean, Min, Max
      - softmax, sigmoid

## Notes

- Model is just the collection of Tensors (Auto-grad tensors, not the raw ones)
- optimizer shares ptr to the model's data (parameters)
- Training Loop:
  - predict (pass some input tensor through the model's layers, get output)
  - loss (take the predictions, and compute loss tensor)
  - backward (the loss is an auto-grad node that can have backward called on it(dumps the grad vaues to each tensor in the model))
  - step (apply, the gradiants to the model's tensors)
  - reset (zero-out the gradiant attached to the model's tensors)
