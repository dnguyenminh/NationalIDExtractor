import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.opencv.core.Core;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NationalIDExtractor {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  // Load OpenCV library
    }

    public static void main(String[] args) throws Exception {
        // Load the input image
        Path imagePath = Paths.get("samples/tai-phoi-can-cuoc-cong-dan-psd.png");
        Image img = ImageFactory.getInstance().fromFile(imagePath);
        Path modelDir = Paths.get("models/national_id_card");
        Block block = ResNetV1.builder().setImageShape(new Shape(3, 256, 256)).setOutSize(10).setNumLayers(18).build();
        Model model = Model.newInstance("resnet");
        model.setBlock(block);
        model.load(modelDir);
        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };
        var predictor = model.newPredictor(translator);
        var detectedObjects = predictor.predict(img);
        // Iterate through the detected objects
        for (Classifications.Classification detectedObject : detectedObjects.items()) {
            if (detectedObject instanceof DetectedObjects.DetectedObject) {
                DetectedObjects.DetectedObject obj = (DetectedObjects.DetectedObject) detectedObject;

                String className = obj.getClassName();
                System.out.println("Detected class: " + className);

                // Check for the "National ID" class (replace with your model's label)
                if ("National ID".equalsIgnoreCase(className)) {
                    BoundingBox bbox = obj.getBoundingBox();

                    if (bbox instanceof Rectangle) {
                        Rectangle rect = (Rectangle) bbox;

                        // Convert normalized bounding box coordinates to pixel coordinates
                        int x = (int) (rect.getX() * img.getWidth());
                        int y = (int) (rect.getY() * img.getHeight());
                        int width = (int) (rect.getWidth() * img.getWidth());
                        int height = (int) (rect.getHeight() * img.getHeight());

                        // Crop the detected region from the image
                        Image croppedImage = img.getSubImage(x, y, width, height);

                        // Save the cropped region
                        Path outputPath = Paths.get("results/tai-phoi-can-cuoc-cong-dan-psd.png");
                        try (OutputStream os = new FileOutputStream(outputPath.toFile())) {
                            croppedImage.save(os, "png");
                        }
                        System.out.println("Cropped National ID saved to: " + outputPath);
                    }
                }
            }
        }
    }
}
