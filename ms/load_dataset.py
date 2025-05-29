from ms.util import *


def load_dataset(dataset_path: str, embodiment_tag: str):
    # 5. load dataset
    dataset = get_load_dataset(dataset_path, embodiment_tag)
    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the 7th data point
    resp = dataset[7]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    return
    # 6. plot the first 100 images
    images_list = []
    video_key = video_modality_keys[0]  # we will use the first video modality

    for i in range(100):
        if i % 10 == 0:
            resp = dataset[i]
            img = resp[video_key][0]
            images_list.append(img)

    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i}")
    plt.tight_layout()  # adjust the subplots to fit into the figure area.
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Robot Dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data/robot_sim.PickNPlace",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="gr1",
        help="Full list of embodiment tags can be found in gr00t.data.schema.EmbodimentTag",
    )
    args = parser.parse_args()
    load_dataset(args.data_path, args.embodiment_tag)
