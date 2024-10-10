from minigrid_custom_env import *
from minigrid_custom_train import *
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, NoDeath

import os
import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

from scipy.spatial.distance import euclidean, cosine

# Collect a dataset of states from the environment
def generate_or_load_state_data(env, num_samples=100000, create_state=False):
    file_path = 'state_data.npz'
    
    # Check if the file exists and is not empty
    if not create_state and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"Loading state data from {file_path}")
        loaded_data = np.load(file_path)
        state_data = loaded_data['state_data']
    else:
        print("Generating new state data...")
        # Generate state data
        state_data = []
        for _ in range(num_samples):
            env.reset()
            state = env.grid.encode()
            state_data.append(state)
        state_data = np.array(state_data)
        # Save the generated state data to a compressed file
        np.savez_compressed(file_path, state_data=state_data)
        print(f"State data saved to {file_path}")

    return state_data


def create_encoder(input_shape, latent_dim=16):
    # Define dimensions
    # input_shape = (7, 7, 3)  # The state shape from your environment
    # latent_dim = 16  # Size of the latent space

    # Encoder
    input_layer = Input(shape=input_shape)
    x = Flatten()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    latent_layer = Dense(latent_dim, activation='relu')(x)

    # Decoder
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(decoder_input)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(np.prod(input_shape), activation='sigmoid')(x)
    output_layer = Reshape(input_shape)(x)

    # Define the autoencoder models
    encoder = Model(inputs=input_layer, outputs=latent_layer, name='encoder')
    decoder = Model(inputs=decoder_input, outputs=output_layer, name='decoder')

    # Combine encoder and decoder into the autoencoder
    autoencoder_input = Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(inputs=autoencoder_input, outputs=decoded, name='autoencoder')

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # Print the model summary for reference
    autoencoder.summary()
    return autoencoder, encoder, decoder


def plot_reconstructed_images(autoencoder, encoder, state_data, env):
    # Select two random states from the dataset
    test_state_1 = state_data[0]
    test_state_1 *= 255
    print(f'test_state_1: {test_state_1}')
    # test_state_2 = state_data[1]
    state_img1 = state_to_image(test_state_1, env)
    # state_img2 = state_to_image(test_state_2, env)


    # # Encode the states
    # encoded_state_1 = encoder.predict(test_state_1.reshape(1, *test_state_1.shape))
    # encoded_state_2 = encoder.predict(test_state_2.reshape(1, *test_state_2.shape))
    
    # # Flatten the encoded states to 1-D vectors
    # encoded_state_1_flat = encoded_state_1.flatten()
    # encoded_state_2_flat = encoded_state_2.flatten()

    # # Calculate the Euclidean distance between the encoded states
    # euclidean_dist = euclidean(encoded_state_1_flat, encoded_state_2_flat)
    # print(f"Euclidean distance between encoded states: {euclidean_dist}")

    # Decode the states
    decoded_state_1 = autoencoder.predict(test_state_1.reshape(1, *test_state_1.shape))
    decoded_state_1 = np.round(decoded_state_1).astype(int)
    print(f'decoded_state_1: {decoded_state_1}')
    # print(f"Decoded state 1 shape: {decoded_state_1.shape},  {type(decoded_state_1)}")
    reconstact_img = state_to_image(decoded_state_1[0], env)
    # decoded_state_2 = autoencoder.predict(test_state_2.reshape(1, *test_state_2.shape))
    # state_img2 = env.grid.decode(decoded_state_2[0])
    
    # Plot the original and reconstructed states
    plt.figure(figsize=(12, 4))

    # Original state 1
    plt.subplot(1, 2, 1)
    plt.title("Original State 1")
    plt.imshow(state_img1)
    plt.axis('off')

    # Reconstructed state 1
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed State 1")
    plt.imshow(reconstact_img)
    plt.axis('off')

    # # Original state 2
    # plt.subplot(1, 3, 3)
    # plt.title("Original State 2")
    # plt.imshow(state_img2)
    # plt.axis('off')

    plt.tight_layout()
    plt.show()

def state_to_image(encoded_grid, env):
    # Decode the state to extract the grid and the agent's position and direction
    grid, vis_mask = Grid.decode(encoded_grid)
    
    # Render the grid with the agent's position and direction
    img = grid.render(tile_size=env.unwrapped.tile_size, agent_pos=env.unwrapped.agent_pos, agent_dir=env.unwrapped.agent_dir)
    
    ## Plot the image using matplotlib
    # plt.imshow(img)
    # plt.axis('off')
    return img

def main():
    # Initialize your custom environment
    env = CustomEnv(size=13, render_mode='rgb_array', difficult_grid=True, max_steps=300, num_objects=5, lava_cells=4)
    env = NoDeath(ImgObsWrapper(ObjObsWrapper(env)), no_death_types=('lava',), death_cost=-2.0)


    # Generate state data
    state_data = generate_or_load_state_data(env, num_samples=100000, create_state=True)

    # Normalize the state data for better training (scaling to [0, 1] range)
    state_data = state_data.astype('float32') / 255.0

    # # Train the autoencoder
    autoencoder, encoder, decoder = create_encoder(input_shape=state_data.shape[1:], latent_dim=32)
    history = autoencoder.fit(state_data, state_data, epochs=10, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

    # Plot training and validation loss over epochs
    # plt.figure(figsize=(8, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.show()

    plot_reconstructed_images(autoencoder, encoder, state_data, env)
    
    env.close()

if __name__ == "__main__":
    main()