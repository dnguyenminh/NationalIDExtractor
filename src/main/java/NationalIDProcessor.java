import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;

public class NationalIDProcessor {
    private static final Logger logger = LoggerFactory.getLogger(NationalIDProcessor.class);

    public static void main(String[] args) throws IOException, MalformedModelException, TranslateException {
        logger.info("Starting National ID Processor");
        try {    // Load the image
            Path imagePath = Paths.get("samples/tai-phoi-can-cuoc-cong-dan-psd.png");
            Image image = ImageFactory.getInstance().fromFile(imagePath);

            // Detect the national ID in the image
            DetectedObjects detection = detectNationalID(image);

            if (detection.items().isEmpty()) {
                logger.warn("No objects detected in the image. Please check if the image contains a national ID and if the model is appropriate for this task.");
                return;
            }

            // Extract the national ID
            Image extractedID = extractNationalID(image, detection);

            if (extractedID == null) {
                logger.warn("Failed to extract the national ID. The detected object might not be a national ID.");
                return;
            }

            // Rotate the ID to the correct position
            Image rotatedID = rotateIDToCorrectPosition(extractedID);
            // Save the processed image
            try (FileOutputStream fileOutputStream = new FileOutputStream("results/tai-phoi-can-cuoc-cong-dan-psd.png")) {
                rotatedID.save(fileOutputStream, "jpg");
            }

        } catch (Exception e) {
            logger.error("An error occurred during processing", e);
        }
        logger.info("Processed national ID saved successfully.");

    }

    private static DetectedObjects detectNationalID(Image image) throws IOException, MalformedModelException, TranslateException {
        // Preprocess the image
        image = preprocessImage(image);

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class) // Define input and output types
                .optApplication(Application.CV.OBJECT_DETECTION) // Specify application
                .optFilter("size", "512") // Filter for size 512
                .optFilter("backbone", "resnet50") // Use resnet50 backbone
                .optFilter("flavor", "v1") // Specify flavor
                .optFilter("dataset", "voc") // Dataset is VOC
                .optGroupId("ai.djl.mxnet") // Specify MXNet engine
                .optArtifactId("ssd") // Specify SSD model
                .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

            DetectedObjects result = predictor.predict(image);

            // Log the detection results
            logger.info("Detection Results:");
            for (DetectedObjects.DetectedObject obj : result.<DetectedObjects.DetectedObject>items()) {
                logger.info("Class: {}, Probability: {}", obj.getClassName(), obj.getProbability());
            }

            return result;
        } catch (ModelNotFoundException e) {
            logger.error("Model not found", e);
            throw new RuntimeException(e);
        }
    }

    private static Image preprocessImage(Image image) throws IOException {
        // Convert DJL Image to BufferedImage
        BufferedImage bufferedImage = (BufferedImage) image.getWrappedImage();

        // Resize the BufferedImage
        BufferedImage resized = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(bufferedImage, 0, 0, 640, 480, null);
        g.dispose();

        // Convert back to DJL Image
        return ImageFactory.getInstance().fromImage(resized);
    }


    private static Image extractNationalID(Image originalImage, Classifications detection) {
        if (detection.items().isEmpty()) {
            logger.warn("No classifications found in the image.");
            return null;
        }

        // Log all classifications
        for (Classifications.Classification c : detection.items()) {
            logger.info("Detected class: {} with probability: {}", c.getClassName(), c.getProbability());
        }

        // For now, we'll just return the entire original image
        // In a real scenario, you'd need to implement logic to determine if any of the
        // classifications correspond to a national ID
        logger.info("Returning full image as no bounding box information is available.");
        return originalImage;
    }

    private static Image rotateIDToCorrectPosition(Image extractedID) {
        // Implement rotation logic here
        // You may need to use image processing techniques or machine learning to determine the correct rotation
        // For simplicity, let's assume the ID is already in the correct position
        return extractedID;
    }
}
