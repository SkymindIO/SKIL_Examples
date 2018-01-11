package ai.skymind.skil.examples.modelserver.inference;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.util.*;

public class NormalizeUciData {

    public static final String DEFAULT_TRAIN_OUTPUT = "/tmp/synthetic_control.data-training-normalized.csv";
    private static final String DEFAULT_TRAIN_SHA1 = "6b5134778cfea12cfe5f463db904bb89aff7a1ec";

    public static final String DEFAULT_TEST_OUTPUT = "/tmp/synthetic_control.data-test-normalized.csv";
    private static final String DEFAULT_TEST_SHA1 = "74177e3b9688344785e7585c596235ae7d84a368";

    private enum Normalizer {
        Standardize (new NormalizerStandardize());

        private DataNormalization normalizer;

        private Normalizer(DataNormalization normalizer) {
            this.normalizer = normalizer;
        }

        public DataNormalization getNormalizer() {
            return normalizer;
        }
    }

    @Parameter(names = "--url", description = "URL to download from", required = false)
    private String downloadUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";

    @Parameter(names = "--normalizer", description = "Normalizer to use", required = false)
    private Normalizer dataNormalizer = Normalizer.Standardize;

    @Parameter(names = "--trainOutput", description = "Where to write data to", required = false)
    private String trainOutputPath = DEFAULT_TRAIN_OUTPUT;

    @Parameter(names = "--testOutput", description = "Where to write data to", required = false)
    private String testOutputPath = DEFAULT_TEST_OUTPUT;

    public void run() throws Exception {
        File trainingOutputFile = new File(trainOutputPath);
        File testOutputFile = new File(testOutputPath);

        if (trainingOutputFile.exists() || testOutputFile.exists()) {
            final FileInputStream fisTrain = new FileInputStream(trainingOutputFile);
            final FileInputStream fisTest = new FileInputStream(testOutputFile);

            final String trainSha1 = DigestUtils.sha1Hex(fisTrain);
            final String testSha1 = DigestUtils.sha1Hex(fisTest);

            fisTrain.close();
            fisTest.close();

            if (!(trainSha1.equals(DEFAULT_TRAIN_SHA1) && testSha1.equals(DEFAULT_TEST_SHA1))) {
                throw new IllegalStateException(
                        String.format("cowardly refusing to overwrite output files (%s, %s)",
                                trainOutputPath, testOutputPath));
            }

            // They match, so tell the user
            System.out.format("output files already found, and have matching checksums, keeping as is\n");
            return;
        }

        System.out.format("downloading from %s\n", downloadUrl);
        System.out.format("writing training output to %s\n", trainOutputPath);
        System.out.format("writing testing output to %s\n", testOutputPath);

        URL url = new URL(downloadUrl);
        String data = IOUtils.toString(url);
        String[] lines = data.split("\n");
        List<INDArray> arrays = new LinkedList<INDArray>();
        List<Integer> labels = new LinkedList<Integer>();

        for (int i=0; i<lines.length; i++) {
            String line = lines[i];
            String[] cols = line.split("\\s+");

            int label = i / 100;
            INDArray array = Nd4j.zeros(1, 60);

            for (int j=0; j<cols.length; j++) {
                Double d = Double.parseDouble(cols[j]);
                array.putScalar(0, j, d);
            }

            arrays.add(array);
            labels.add(label);
        }

        // Shuffle with **known** seed
        Collections.shuffle(arrays, new Random(12345));
        Collections.shuffle(labels, new Random(12345));

        INDArray trainData = Nd4j.zeros(450, 60);
        INDArray testData = Nd4j.zeros(150, 60);

        for (int i=0; i<arrays.size(); i++) {
            INDArray arr = arrays.get(i);

            if (i < 450) { // Training
                trainData.putRow(i, arr);
            } else { // Test
                testData.putRow(i-450, arr);
            }
        }

        DataSet trainDs = new DataSet(trainData, trainData);
        DataSetIterator trainIt = new ListDataSetIterator(trainDs.asList());

        DataSet testDs = new DataSet(testData, testData);
        DataSetIterator testIt = new ListDataSetIterator(testDs.asList());

        // Fit normalizer on training data only!
        DataNormalization normalizer = dataNormalizer.getNormalizer();
        normalizer.fit(trainIt);

        // Print out basic summary stats
        switch (normalizer.getType()) {
            case STANDARDIZE:
                System.out.format("Normalizer - Standardize:\n  mean=%s\n  std= %s\n",
                        ((NormalizerStandardize)normalizer).getMean(),
                        ((NormalizerStandardize)normalizer).getStd());
        }

        // Use same normalizer for both
        trainIt.setPreProcessor(normalizer);
        testIt.setPreProcessor(normalizer);

        String trainOutput = toCsv(trainIt, labels.subList(0, 450), new int[]{1, 60});
        String testOutput = toCsv(testIt, labels.subList(450, 600), new int[]{1, 60});

        FileUtils.write(trainingOutputFile, trainOutput);
        System.out.format("wrote normalized training file to %s\n", trainingOutputFile);

        FileUtils.write(testOutputFile, testOutput);
        System.out.format("wrote normalized test file to %s\n", testOutputFile);

    }

    private String toCsv(DataSetIterator it, List<Integer> labels, int[] shape) {
        if (it.numExamples() != labels.size()) {
            throw new IllegalStateException(
                    String.format("numExamples == %d != labels.size() == %d",
                            it.numExamples(), labels.size()));
        }

        StringBuffer sb = new StringBuffer();
        int l = 0;

        while (it.hasNext()) {
            INDArray features = it.next(1).getFeatures();

            if (!(Arrays.equals(features.shape(), shape))) {
                throw new IllegalStateException(String.format("wrong shape: got %s, expected",
                        Arrays.toString(features.shape()), Arrays.toString(shape)));
            }

            // Prepend the label
            sb.append(labels.get(l)).append(": ");
            l++;

            for (int i=0; i<features.columns(); i++) {
                sb.append(features.getColumn(i));

                if (i < features.columns()-1) {
                    sb.append(", ");
                }
            }

            sb.append("\n");
        }

        return sb.toString();
    }


    public static void main(String... args) throws Exception {
        NormalizeUciData m = new NormalizeUciData();
        JCommander.newBuilder()
                .addObject(m)
                .build()
                .parse(args);

        m.run();
    }
}
