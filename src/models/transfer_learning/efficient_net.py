import ssl
from transfer_learning import TransferLearning


def main(train=False):
    # For downloading models from PyTorch Vision
    ssl._create_default_https_context = ssl._create_unverified_context
    if train:
        # Transfer Learning
        transfer_learning = TransferLearning(model_type="efficientnet", size=(224, 224), do_oversampling=False, do_loss_weights=False)
        transfer_learning.read_metadata_dataframe(path="data/ham10000/metadata.csv")
        transfer_learning.perform_dataset_split()
        transfer_learning.create_dataloaders()
        transfer_learning.prepare_model()
        transfer_learning.train_model()
        transfer_learning.test_model()
        transfer_learning.dump_metrics()
    else:
        transfer_learning = TransferLearning(model_type="efficientnet", size=(224, 224), do_oversampling=False, do_loss_weights=False)
        transfer_learning.read_metadata_dataframe(path="data/ham10000/metadata.csv")
        transfer_learning.perform_dataset_split()
        transfer_learning.create_dataloaders()
        transfer_learning.prepare_model()
        transfer_learning.load_model()
        transfer_learning.test_model()
    print("Finished!")


if __name__ == "__main__":
    main()
