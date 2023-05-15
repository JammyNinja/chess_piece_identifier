#TODO

#def get_test_image_and_label()


#def load_data()
#return X_train, X_val, X_test


def get_test_image_and_label_as_numpy(X_batches_in):

    for image_batch, label_batch in X_batches_in:
        for image, label in zip(image_batch, label_batch):
            return image.numpy(), label.numpy()

#def show_class_distributions():
#plot train and test spread of classes (#order the x axis)

#def create_baseline_dict():
