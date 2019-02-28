package io.skymind.skil.tutorial;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;

import java.io.File;

public class CreateInferenceTransformDescription {
    @Parameter(names = "--input", description = "Path to data analysis file.")
    private File dataAnalysis;

    @Parameter(names = "--output", description = "Path to output transform process JSON.")
    private String outputPath;

    public static void main(String...args) throws Exception {
        new CreateInferenceTransformDescription().entryPoint(args);
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

        DataAnalysis analysis = DataAnalysis.fromJson(FileUtils.readFileToString(dataAnalysis));

        TransformProcess tp = IrisData.inferenceTransform(analysis);

        FileUtils.writeStringToFile(new File(outputPath + File.separator + "iris-inference-transform.json"), tp.toJson());
    }
}
