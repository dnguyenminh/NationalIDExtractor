import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class ResizeExample {

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        NDManager manager = NDManager.newBaseManager();
        Path originalFolder = Paths.get("original");
        Path datasetFolder = Paths.get("dataset");
        // Define target dimensions
        int targetWidth = 300;
        int targetHeight = 300;

        resizeAndSave(manager, targetWidth, targetHeight, 1, originalFolder, datasetFolder, true);
    }

    private static void resizeAndSave(NDManager manager, int targetWidth, int targetHeight, int interpolation, Path originalFolder, Path datasetFolder, boolean addPadding) throws IOException, ExecutionException, InterruptedException {
        if (originalFolder.toFile().exists()) {
            if (originalFolder.toFile().isDirectory()) {
                for (File imageFile : Objects.requireNonNull(originalFolder.toFile().listFiles())) {
                    resizeAndSave(manager, targetWidth, targetHeight, interpolation,
                            originalFolder.resolve(imageFile.getName()),
                            datasetFolder.resolve(imageFile.getName()), addPadding);
                }
            } else {
                datasetFolder.toFile().mkdirs();
                resizeAndSaveFile(manager, targetWidth, targetHeight, interpolation, originalFolder, datasetFolder, addPadding);
            }
        }
    }

    private static void resizeAndSaveFile(NDManager manager, int targetWidth, int targetHeight, int interpolation, Path imagePath, Path outputFilePath, boolean addPadding) throws IOException, ExecutionException, InterruptedException {
        // Load the image
        Image img = ImageFactory.getInstance().fromFile(imagePath);

        // Convert image to NDArray
        NDArray originalNDArray = img.toNDArray(manager);

        // Get original image dimensions
        Shape shape = originalNDArray.getShape();
        long originalWidth = shape.get(1); // Width
        long originalHeight = shape.get(0); // Height

        // Preserve aspect ratio
        double aspectRatio = (double) originalWidth / originalHeight;
        int tempWidth = targetWidth;
        int tempHeight = targetHeight;

        if (originalWidth > originalHeight) {
            tempHeight = (int) (targetWidth / aspectRatio);
        } else {
            tempWidth = (int) (targetHeight * aspectRatio);
        }
        final int newWidth = tempWidth;
        final int newHeight = tempHeight;

        // Resize the NDArray using the calculated new dimensions
        NDArray resizedNDArray = originalNDArray.getNDArrayInternal().resize(newWidth, newHeight, interpolation); // Bilinear interpolation

        // Create a canvas for the padded image
        final NDArray paddedNDArray = manager.zeros(new Shape(targetHeight, targetWidth, 3)); // 3 channels for RGB

        if (addPadding) {
            // Calculate padding offsets
            int offsetX = (targetWidth - newWidth) / 2;
            int offsetY = (targetHeight - newHeight) / 2;

            // Parallelize the padding operation for improved performance
            ForkJoinPool pool = new ForkJoinPool();
            pool.submit(() -> {
                // Using parallel processing for padding the image
                for (int y = 0; y < newHeight; y++) {
                    for (int x = 0; x < newWidth; x++) {
                        // Access the pixel at (y, x) for all channels (R, G, B)
                        NDIndex pixelIndex = new NDIndex(y, x, 0); // Resize image pixel for Red channel
                        NDArray pixel = resizedNDArray.get(pixelIndex).toType(DataType.FLOAT32, true); // Convert to float32

                        // Calculate the target position in the padded canvas
                        NDIndex paddedIndex = new NDIndex(offsetY + y, offsetX + x, 0); // Red channel
                        paddedNDArray.set(paddedIndex, pixel);

                        pixelIndex = new NDIndex(y, x, 1); // Green channel
                        pixel = resizedNDArray.get(pixelIndex).toType(DataType.FLOAT32, true);
                        paddedIndex = new NDIndex(offsetY + y, offsetX + x, 1); // Green channel
                        paddedNDArray.set(paddedIndex, pixel);

                        pixelIndex = new NDIndex(y, x, 2); // Blue channel
                        pixel = resizedNDArray.get(pixelIndex).toType(DataType.FLOAT32, true);
                        paddedIndex = new NDIndex(offsetY + y, offsetX + x, 2); // Blue channel
                        paddedNDArray.set(paddedIndex, pixel);
                    }
                }
            }).get();
        }

//        // Convert the padded image back to uint8
//        paddedNDArray = paddedNDArray.toType(DataType.UINT8, true);

        // Convert resized NDArray back to Image
        Image resizedImg = ImageFactory.getInstance().fromNDArray(paddedNDArray.toType(DataType.UINT8, true));

        // Save the resized image to an output stream
        try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            resizedImg.save(outputStream, "jpg");
            Files.write(outputFilePath, outputStream.toByteArray());
        }

        System.out.println("Resized image saved to: " + outputFilePath);
    }
}
