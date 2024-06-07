import ssl
from transfer_learning import TransferLearning


def main(train=False, sampling=False, class_weights=False):
    # For downloading models from PyTorch Vision
    ssl._create_default_https_context = ssl._create_unverified_context
    # Transfer Learning
    transfer_learning = TransferLearning(model_type="resnet", size=(224, 224), do_oversampling=sampling, do_loss_weights=class_weights)
    transfer_learning.read_metadata_dataframe(path="data/ham10000/metadata.csv")
    transfer_learning.perform_dataset_split()
    transfer_learning.create_dataloaders()
    transfer_learning.prepare_model()
    if train:
        transfer_learning.train_model()
        transfer_learning.load_model()
        transfer_learning.test_model()
        transfer_learning.dump_metrics()
    else:
        transfer_learning.load_model()
        transfer_learning.test_model()
    print("Finished!")


if __name__ == "__main__":
    main(train=True, sampling=False, class_weights=True)
