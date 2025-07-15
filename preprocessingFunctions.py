# Holds all the functions for preprocessing 
import tensorflow as tf

## Function to decode and normalize the images and standardize to RGB
image_size = [224,224]

def decode_image(image_data):
    """Function to decode the image from the .tfrec"""
    ## Converts the raw JPEG file bytes into a 3D tensor, channels=3  indicates RGB, create the shape (height, width, 3), the output is a uint tensor with values 0-255
    image = tf.image.decode_jpeg(image_data, channels=3)
    
    ## Resize all images to the image size specified above [512,512], using bilinear method to smooth the images for efficient processing
    image = tf.image.resize(image, image_size, method = "bilinear")
    
    ## Converts the uint to float32, then normalizes the inputs by dividing by the number of pixel values 255
    image = tf.cast(image, tf.float32) / 255.0
    
    ## Takes the image size defined above this function and reshapes it to be the image size [height, width, 3]
    image = tf.reshape(image, [*image_size,3])
    return image

## Function to return an image, label pair for the training and validation sets
def read_labeled_tfrec(input_example):
    """Read and parse the labeled .tfrec"""
    ## Tells Tensorflow how to interpret the binary .tfrec data, "image" tells TF to expect binary jppeg bytes, "class" tells TF to expect integer labels (the flower labels)
    labeled_tfrec_format = { 
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64)
    }
    ## Parses the input_example using the format specified above
    ## Takes raw binary data (input_example) and uses the labeled_tfrec_format to return a dictionary {image bytes, flower label}
    input_example = tf.io.parse_single_example(input_example, labeled_tfrec_format)
    
    ## Process the image - Takes the JPEG bytes from the example image, and normalizes them to a [512,512,3] tensor using the decode_image function
    image = decode_image(input_example["image"])
    
    ## Process the label - input_example['class'] is the flower class ID
    label = tf.cast(input_example['class'], tf.int32)
    return image, label

## Function to return an image without labels for the test set
def read_unlabeled_tfrec(input_example):
    """Read and parse the unlabeled .tfrec"""
    unlabeled_tfrec_format = { 
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string)
    }    
    input_example = tf.io.parse_single_example(input_example, unlabeled_tfrec_format)
    
    ## Process the image - Takes the JPEG bytes from the example image, and normalizes them to a [512,512,3] tensor using the decode_image function
    image = decode_image(input_example["image"])
    
    ## Process the label - input_example['id'] is the image ID
    image_id = input_example['id']
    return image, image_id

def load_dataset(filenames, labeled=True):
    """Load TFRecord dataset from filenames"""
    ## Creates a dataset that reads the files, AUTOTUNE processes them simultneously and TF optimizes the number of readers
    ## Creates a dataset of the raw binary .tfrec examples
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)

    ## dataset.map applies the read_labeled_tfrec function to each example, AUTOTUNE processes them simultneously and TF optimizes the number of readers
    ## Transforms the raw data to (image_tensor, label_int) pairs
    if labeled:
        dataset = dataset.map(read_labeled_tfrec, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(read_unlabeled_tfrec, num_parallel_calls=tf.data.AUTOTUNE)       
    return dataset

# ## Combine all four image sizes into standardized datasets
# image_size_folders = ['tfrecords-jpeg-192x192', 'tfrecords-jpeg-224x224', 'tfrecords-jpeg-331x331', 'tfrecords-jpeg-512x512']

# ## Initialize empty lists to collect all filenames from all image sizes
# all_training_filenames = []
# all_validation_filenames = []
# all_test_filenames = []

# ## Collect filenames from all four image size folders
# for folder in image_size_folders:
#     train_files = tf.io.gfile.glob(f"./tpu-getting-started/{folder}/train/*.tfrec")
#     val_files = tf.io.gfile.glob(f"./tpu-getting-started/{folder}/val/*.tfrec")
#     test_files = tf.io.gfile.glob(f"./tpu-getting-started/{folder}/test/*.tfrec")
    
#     all_training_filenames.extend(train_files)
#     all_validation_filenames.extend(val_files)
#     all_test_filenames.extend(test_files)
    

# ## Creates standardized datasets with images from all four sizes
# training_filenames = all_training_filenames
# validation_filenames = all_validation_filenames
# test_filenames = all_test_filenames

# ## Load and create the standardized training dataset
# train_dataset = load_dataset(training_filenames, labeled=True)

# ## Load and create the standardized validation dataset  
# validation_dataset = load_dataset(validation_filenames, labeled=True)

# ## Load and create the standardized test dataset
# test_dataset = load_dataset(test_filenames, labeled=False)

# ## Shuffling the training and validation sets

# ## Sets random seed for reproducability
# tf.random.set_seed(42)

# ## Set shuffle buffer to set how many are shuffled at once
# shuffle_buffer = 500

# ## Shuffle the training set
# train_dataset = train_dataset.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=False)

# ## Shuffle the validation set
# validation_dataset = validation_dataset.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=False)


# Class to simplify usage of these functions
class TFRecordPipeline:
    def __init__(self, base_path="/content/drive/My Drive/Flower Classification Project/Data", image_size_folders=None, shuffle_buffer=500, seed=42):
        self.base_path = base_path
        self.image_size_folders = image_size_folders or [
            'tfrecords-jpeg-192x192',
            'tfrecords-jpeg-224x224',
            'tfrecords-jpeg-331x331',
            'tfrecords-jpeg-512x512'
        ]
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        tf.random.set_seed(seed)

    def collect_filenames(self):
        """Collect all .tfrec file paths from the image size folders, with file counts"""
        all_training_filenames = []
        all_validation_filenames = []
        all_test_filenames = []

        for folder in self.image_size_folders:
            train_files = tf.io.gfile.glob(f"{self.base_path}/{folder}/train/*.tfrec")
            val_files = tf.io.gfile.glob(f"{self.base_path}/{folder}/val/*.tfrec")
            test_files = tf.io.gfile.glob(f"{self.base_path}/{folder}/test/*.tfrec")

            all_training_filenames.extend(train_files)
            all_validation_filenames.extend(val_files)
            all_test_filenames.extend(test_files)

            print(f"From {folder}:")
            print(f"  Training files: {len(train_files)}")
            print(f"  Validation files: {len(val_files)}")
            print(f"  Test files: {len(test_files)}")

        print(f"\nCOMBINED TOTALS:")
        print(f"Training files: {len(all_training_filenames)}")
        print(f"Validation files: {len(all_validation_filenames)}")
        print(f"Test files: {len(all_test_filenames)}")

        return all_training_filenames, all_validation_filenames, all_test_filenames

    def get_datasets(self):
        """Load, shuffle, and return training, validation, and test datasets"""
        training_filenames, validation_filenames, test_filenames = self.collect_filenames()

        print("\nLoading datasets...")

        ## Load and create the standardized training dataset
        train_dataset = load_dataset(training_filenames, labeled=True)
        print(f"Training dataset created from {len(training_filenames)} files")

        ## Load and create the standardized validation dataset
        validation_dataset = load_dataset(validation_filenames, labeled=True)
        print(f"Validation dataset created from {len(validation_filenames)} files")

        ## Load and create the standardized test dataset
        test_dataset = load_dataset(test_filenames, labeled=False)
        print(f"Test dataset created from {len(test_filenames)} files")

        ## Shuffle the training set
        train_dataset = train_dataset.shuffle(self.shuffle_buffer, seed=self.seed, reshuffle_each_iteration=False)

        ## Shuffle the validation set
        validation_dataset = validation_dataset.shuffle(self.shuffle_buffer, seed=self.seed, reshuffle_each_iteration=False)

        return train_dataset, validation_dataset, test_dataset
