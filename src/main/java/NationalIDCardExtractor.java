import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class NationalIDCardExtractor {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  // Load OpenCV library
    }

    private Model model;
    private Predictor<Image, NDArray> predictor;

    public NationalIDCardExtractor(String name, Path modelFolderPath) throws IOException, ModelException {
        // Load the trained model
        model = Model.newInstance(name);
        model.load(modelFolderPath);
        predictor = model.newPredictor(new ImageTranslator());
    }

    public NDArray extractFeatures(String imagePath) throws Exception {
        // Load and process the image
        Mat mat = ResizeExampleByOpenCV.resize(Paths.get(imagePath), 256, 156, true);
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);

        Image img = ImageFactory.getInstance().fromInputStream(new ByteArrayInputStream(matOfByte.toArray()));
        NDArray result = predictor.predict(img);
        return result;
    }

    public static void main(String[] args) {
        try {
            NationalIDCardExtractor extractor = new NationalIDCardExtractor(
                    "resnet", Paths.get("models/national_id_card"));

            // Path to the image you want to classify
            String imagePath = "path/to/national_id_image.jpg";
            NDArray result = extractor.extractFeatures(imagePath);

            // Output the results
            System.out.println("Extracted features: " + result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class ImageTranslator implements ai.djl.translate.Translator<Image, NDArray> {
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        // Convert image to NDArray directly
        NDArray ndArray = input.toNDArray(ctx.getNDManager());

//        // Resize the NDArray to 256x256 (adjust to three channels)
//        ndArray = ndArray.getNDArrayInternal().resize(256, 256, 1);

        // Normalize the NDArray with mean and standard deviation
        ndArray = ndArray.sub(0.5f).div(0.5f); // Alternatively, you can customize for your use case

        // Add batch dimension to the NDArray, making shape [1, 3, 256, 256]
        ndArray = ndArray.expandDims(0); // Expand dims to add a batch size of 1

        return new NDList(ndArray);
    }


    @Override
    public NDArray processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return list.singletonOrThrow(); // Return the first item in the NDList
    }

    // Optionally implement getBatchifier if needed, otherwise return null
    @Override
    public Batchifier getBatchifier() {
        return null; // No batching necessary for individual images
    }

}
