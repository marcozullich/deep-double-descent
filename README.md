# deep-double-descent

Minimal implementation for deep double descent with artificial neural networks. Still work in progress.


## dependencies

Requires PyTorch and matplotlib


## how to run

For KMNIST:

`python double_desc.py --num_epochs=10 --net cnn --dataset kmnist --save_file <name_of_save_file.pt>`

To visualize the results:

`python plot_results.py --file <name_of_save_file.pt> [--image <name_of_image_file.png>]`
