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
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.misc.StringToWritablesFunction;

import java.io.File;
import java.util.List;

public class AnalyzeIrisData {
    @Parameter(names = "--input", description = "Training data directory.")
    private String trainDataPath;

    @Parameter(names = "--output", description = "The directory to put the analysis files.")
    private String outputPath;

    public static void main(String...args) throws Exception {
        new AnalyzeIrisData().entryPoint(args);
    }

    private void entryPoint(String... args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            System.out.println(e);
            jcmdr.usage();
            System.exit(1);
        }

        Schema schema = IrisData.SCHEMA;

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("Iris Data Analysis");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> stringData = sc.textFile(trainDataPath);

        //We first need to parse this comma-delimited (CSV) format; we can do this using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        int maxHistogramBuckets = 10;
        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(schema, parsedInputData, maxHistogramBuckets);

        System.out.println(dataAnalysis);

        String json = dataAnalysis.toJson();
        File out = new File(outputPath + File.separator + "iris-analysis.json");
        FileUtils.writeStringToFile(out, json);

        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, new File(outputPath + File.separator + "iris-analysis.html"));

        sc.stop();
    }
}
