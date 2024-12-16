import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

public class ResizeExampleByOpenCV {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  // Load OpenCV library
    }

    public static void main(String[] args) throws Exception {
        Path originalFolder = Paths.get("original");
        Path datasetFolder = Paths.get("dataset");

        // Define target dimensions
        int targetWidth = 300;
        int targetHeight = 300;

        resizeAndSave(originalFolder, datasetFolder, targetWidth, targetHeight, true);
    }

    public static void resizeAndSave(Path originalFolder, Path datasetFolder, int targetWidth, int targetHeight, boolean addPadding) throws Exception {
        if (originalFolder.toFile().exists()) {
            if (originalFolder.toFile().isDirectory()) {
                for (File imageFile : Objects.requireNonNull(originalFolder.toFile().listFiles())) {
                    datasetFolder.toFile().mkdirs();
                    resizeAndSave(
                            originalFolder.resolve(imageFile.getName()),
                            datasetFolder.resolve(imageFile.getName()), targetWidth, targetHeight, addPadding);
                }
            } else {
                resizeAndSaveFile(originalFolder, datasetFolder, targetWidth, targetHeight, addPadding);
            }
        }
    }

    private static void resizeAndSaveFile(Path imagePath, Path outputFilePath, int targetWidth, int targetHeight, boolean addPadding) throws Exception {
//        // Read image using OpenCV
//        Mat image = Imgcodecs.imread(imagePath.toString());
//
//        if (image.empty()) {
//            System.out.println("Failed to load image: " + imagePath);
//            return;
//        }
//
//        // Get the original aspect ratio
//        double aspectRatio = (double) image.cols() / image.rows();
//        int newWidth = targetWidth;
//        int newHeight = targetHeight;
//
//        // Resize the image while maintaining the aspect ratio
//        if (aspectRatio > 1) {
//            // Landscape image (wider than tall)
//            newHeight = (int) (targetWidth / aspectRatio);
//        } else {
//            // Portrait image (taller than wide)
//            newWidth = (int) (targetHeight * aspectRatio);
//        }
//
//        Mat resizedImage = new Mat();
//        Size newSize = new Size(newWidth, newHeight);
//        Imgproc.resize(image, resizedImage, newSize);
//
//        if (addPadding) {
//            // Calculate padding
//            int paddingTop = (targetHeight - resizedImage.rows()) / 2;
//            int paddingBottom = targetHeight - resizedImage.rows() - paddingTop;
//            int paddingLeft = (targetWidth - resizedImage.cols()) / 2;
//            int paddingRight = targetWidth - resizedImage.cols() - paddingLeft;
//
//            // Add black padding around the resized image
//            Mat paddedImage = new Mat();
//            Core.copyMakeBorder(resizedImage, paddedImage, paddingTop, paddingBottom, paddingLeft, paddingRight, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
//
//            // Save the final image with padding
////            Imgcodecs.imwrite(outputFilePath.toString(), paddedImage);
////            MatOfByte matOfByte = new MatOfByte();
////            Imgcodecs.imencode(".jpg", paddedImage, matOfByte);
//            System.out.println("Resized image with padding saved to: " + outputFilePath);
//        } else {
//            // Save the resized image without padding
//            Imgcodecs.imwrite(outputFilePath.toString(), resizedImage);
//            System.out.println("Resized image saved to: " + outputFilePath);
//        }
        Mat mat = resize(imagePath, targetWidth, targetHeight, addPadding);
        Imgcodecs.imwrite(outputFilePath.toString(), mat);
    }

    public static Mat resize(Path imagePath, int targetWidth, int targetHeight, boolean addPadding) throws Exception {
        // Read image using OpenCV
        Mat image = Imgcodecs.imread(imagePath.toString());

//        if (image.empty()) {
//            System.out.println("Failed to load image: " + imagePath);
//            return;
//        }

        // Get the original aspect ratio
        double aspectRatio = (double) image.cols() / image.rows();
        int newWidth = targetWidth;
        int newHeight = targetHeight;

        // Resize the image while maintaining the aspect ratio
        if (aspectRatio > 1) {
            // Landscape image (wider than tall)
            newHeight = (int) (targetWidth / aspectRatio);
        } else {
            // Portrait image (taller than wide)
            newWidth = (int) (targetHeight * aspectRatio);
        }

        Mat resizedImage = new Mat();
        Size newSize = new Size(newWidth, newHeight);
        Imgproc.resize(image, resizedImage, newSize);

        if (addPadding) {
            // Calculate padding
            int paddingTop = (targetHeight - resizedImage.rows()) / 2;
            int paddingBottom = targetHeight - resizedImage.rows() - paddingTop;
            int paddingLeft = (targetWidth - resizedImage.cols()) / 2;
            int paddingRight = targetWidth - resizedImage.cols() - paddingLeft;

            // Add black padding around the resized image
            Mat paddedImage = new Mat();
            Core.copyMakeBorder(resizedImage, paddedImage, paddingTop, paddingBottom, paddingLeft, paddingRight, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

            // Save the final image with padding
//            Imgcodecs.imwrite(outputFilePath.toString(), paddedImage);
//            MatOfByte matOfByte = new MatOfByte();
//            Imgcodecs.imencode(".jpg", paddedImage, matOfByte);
            return paddedImage;
        } else {
            // Save the resized image without padding
            return resizedImage;
        }
    }
}
