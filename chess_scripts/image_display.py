#TODO
#softcode class names, flag path as env (os.environ() ?)
#path to data also?

import matplotlib.pyplot as plt
import os
from math import ceil

class ImageDisplay:
    def __init__(self):
        print(f"Hello {__name__} from init func")
        self.class_names = ["bishop", "knight", "pawn", "queen", "rook"]

    def plot_image(self, ax, image=None, title=""):
        #scale if necessary
        if image.max() > 1:
            print("rescaling by /255")
            image = image/ 255

        ax.set_title(title)
        ax.axis('off')
        ax.imshow(image)
        return ax

    def show_batch_images(self, image_batch_in, N=8, num_cols=4):

        #see how many batches we will need
        batch_size = 0
        batches = 1
        for batch, labels in image_batch_in:
            batch_size = batch.shape[0]
        if N > batch_size:
            batches = ceil(N/batch_size)

        #collect images from batch
        images_to_show, labels_to_show = [],[]
        for image_batch, label_batch in image_batch_in.take(batches).as_numpy_iterator():
            #print(image.shape)
            for image, label in zip(image_batch, label_batch):
                if len(labels_to_show) < N:
                    images_to_show.append(image)
                    labels_to_show.append(label)

        num_cols = 4
        num_rows = ceil(N/num_cols)

        #now display them
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*3,num_rows*2))
        fig.suptitle(f"Displaying {N} training images")

        for i, image in enumerate(labels_to_show):
            x = i % num_cols
            y = i // num_cols
            axs[y,x].axis('off')
            axs[y,x].imshow(images_to_show[i]/255);
            axs[y,x].set_title(f"{self.class_names[labels_to_show[i] ]}")

        return fig, axs

    def inspect_all_images(self, X_in, ax=None, start=0):
        """ Cycles through the image dataset passed in, asking for input() after each image
        if any input given, then that text is appended to flagged_images.txt along with the 'class/file_name.jpg'
        """
        fig, this_ax = plt.subplots(figsize=(10,6))

        for i, (image,label) in enumerate( X_in.unbatch().as_numpy_iterator() ):
            if i < start:
                continue

            #get/set meta
            title = f"image #{i} : {self.class_names[label]}"
            file = "/".join(X_in.file_paths[i].split('/')[3:])

            #plot and show the ax
            self.plot_image(ax=plt.gca(), image=image, title=title)
            plt.show()

            #get and store feedback
            comment = input(str(X_in.file_paths[i] + "\n"))
            if comment:
                with open('../data/flagged_images.txt', 'a') as flag_file:
                    flag_file.write(f"{file} #{i} : {comment} \n")

            #clear axis and repeat
            plt.cla()
        return None

    def delete_flagged_files(self, flagfile_path='../data/flagged_images.txt'):
        """ cycles through the flagged_images.txt , and uses the first part of line
            deletes it if it exists
        """
        paths_to_delete = []
        path_to_data = "../data/chess_pieces_images/"

        with open(flagfile_path, 'r') as flag_file:
            for line in flag_file:
                file_to_delete = line.split(" ")[0]
                path_to_delete = os.path.join(path_to_data, file_to_delete)
                paths_to_delete.append(path_to_delete)

        del_count = 0
        for path in paths_to_delete:
            if os.path.isfile(path):
                os.remove(path)
                del_count += 1

        #print(f"deleted {len(paths_to_delete)} files")
        out_str = f"deleted {del_count} out of {len(paths_to_delete)} in list"
        return out_str

if __name__ == "__main__":
    print("image_display_file being run from terminal")
    id = ImageDisplay()
