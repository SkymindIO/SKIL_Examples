package io.skymind.skil.tutorial;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationUtils;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.List;

public class TrainModel {
    @Parameter(names = "--train", description = "Directory for the training data.")
    private String trainDirectory;

    @Parameter(names = "--test", description = "Directory for the test data.")
    private String testDirectory;

    @Parameter(names = "--analysis", description = "Path to the analysis json file.")
    private File analysisFile;

    @Parameter(names = "--epochs", description = "Number of epochs to train.")
    private int epochs;

    // Could also load from json file.
    private MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .iterations(1)
            .seed(42)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .list(
                    new DenseLayer.Builder()
                            .nIn(IrisData.INPUT_COUNT)
                            .nOut(10)
                            .activation(Activation.TANH)
                            .build(),
                    new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nIn(10)
                            .nOut(IrisData.LABEL_COUNT)
                            .build())
            .pretrain(false)
            .backprop(true)
            .build();

    public static void main(String... args) throws Exception {
        TrainModel instance = new TrainModel();
        JCommander jcmdr = new JCommander(instance);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            System.out.println(e);
            jcmdr.usage();
            System.exit(1);
        }

        instance.train();
    }

    private void train() throws Exception {
        Schema schema = IrisData.SCHEMA;

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Iris Data Analysis");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> trainData = sc.textFile(trainDirectory);
        JavaRDD<String> testData = sc.textFile(testDirectory);

        JavaRDD<List<Writable>> parsedTrainData = trainData.map(new StringToWritablesFunction(new CSVRecordReader()));
        JavaRDD<List<Writable>> parsedTestData = testData.map(new StringToWritablesFunction(new CSVRecordReader()));

        DataAnalysis analysis = DataAnalysis.fromJson(FileUtils.readFileToString(analysisFile));
        TransformProcess tp = IrisData.trainTransform(analysis);

        JavaRDD<List<Writable>> normalizedTrainData = SparkTransformExecutor.execute(parsedTrainData, tp);
        JavaRDD<List<Writable>> normalizedTestData = SparkTransformExecutor.execute(parsedTestData, tp);

        List<DataSet> trainSet = normalizedTrainData.map(new DataVecDataSetFunction(IrisData.LABEL_INDEX, IrisData.LABEL_COUNT, false)).collect();
        List<DataSet> testSet = normalizedTestData.map(new DataVecDataSetFunction(IrisData.LABEL_INDEX, IrisData.LABEL_COUNT, false)).collect();

        ListDataSetIterator train = new ListDataSetIterator(trainSet);
        ListDataSetIterator test = new ListDataSetIterator(testSet);

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(
                        new BestScoreEpochTerminationCondition(0.01),
                        new MaxEpochsTerminationCondition(epochs))
                .scoreCalculator(new DataSetLossCalculator(test, false))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, train);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        MultiLayerNetwork model = result.getBestModel();

        Evaluation eval = new Evaluation(IrisData.LABEL_COUNT);
        ROCMultiClass roc = new ROCMultiClass(10);

        test.reset();
        while (test.hasNext()) {
            DataSet ds = test.next();
            INDArray output = model.output(ds.getFeatures());
            INDArray labels = ds.getLabels();
            eval.eval(labels, output);
            roc.eval(labels, output);
        }

        System.out.println(eval.stats());

        FileUtils.writeStringToFile(new File("irismodel.eval"), eval.stats());
        EvaluationTools.exportRocChartsToHtmlFile(roc, new File("irismodel.roc.html"));
        ModelSerializer.writeModel(model, "irismodel.zip", true);

        sc.stop();
    }
}