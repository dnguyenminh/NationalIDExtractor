import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TrainingNationalIDCardDetector {
    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 16;

        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get("dataset"))
                .addTransform(new Resize(256, 256, Image.Interpolation.BILINEAR)) // Resizing the images to 256x256
                .addTransform(new ToTensor())  // Normalizing data
                .setSampling(batchSize, true)
                .build();

        dataset.prepare(new ProgressBar());

        // Load or Build a more complex model, for instance, a pretrained ResNet
        Block block = ResNetV1.builder().setImageShape(new Shape(3, 256, 256)).setOutSize(10).setNumLayers(18).build();
        Model model = Model.newInstance("resnet");
        model.setBlock(block);

        // Configure training parameters
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 3, 256, 256));

        int epoch = 10;  // Adjust the number of epochs
        EasyTrain.fit(trainer, epoch, dataset, null);

        Path modelDir = Paths.get("models/national_id_card");
        Files.createDirectories(modelDir);

        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(modelDir, "national_id_card");

        System.out.println(model);
    }
}
