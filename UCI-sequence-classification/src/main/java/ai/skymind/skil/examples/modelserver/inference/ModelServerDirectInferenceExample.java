package ai.skymind.skil.examples.modelserver.inference;

import ai.skymind.skil.examples.modelserver.inference.model.Inference;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class ModelServerDirectInferenceExample {

    @Parameter(names = "--inference", description = "Endpoint for Inference", required = true)
    private String inferenceEndpoint;

    @Parameter(names = "--input", description = "CSV input file", required = true)
    private String inputFile;

    @Parameter(names = "--sequential", description = "If this transform a sequential one", required = false)
    private boolean isSequential = false;

    @Parameter(names = "--textAsJson", description = "Parse text/plain as JSON", required = false, arity = 1)
    private boolean textAsJson = true;

    public void run() throws Exception {
        final File file = new File(inputFile);

        if (!file.exists() || !file.isFile()) {
            System.err.format("unable to access file %s\n", inputFile);
            System.exit(2);
        }

        // Open file
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

        SkilClient skilClient = new SkilClient(textAsJson);

        // Read each line
        String line = null;
        while ((line = br.readLine()) != null) {
            // Check if label indicator is up front
            String label = null;
            if (line.matches("^\\d:\\s.*")) {
                label = line.substring(0, 1);
            }

            // Just in case
            line = StringUtils.removePattern(line, "^\\d:\\s");
            String[] fields = line.split(",");

            // Maybe strip quotes
            for (int i = 0; i < fields.length; i++) {
                final String field = fields[i];
                if (field.matches("^\".*\"$")) {
                    fields[i] = field.substring(1, field.length() - 1);
                }
            }

            int[] shape = (isSequential) ?
                    new int[] { 1, 1, fields.length} :
                    new int[] { 1, fields.length};

            INDArray array = Nd4j.create(shape);

            for (int i=0; i<fields.length; i++) {
                // TODO: catch NumberFormatException
                Double d = Double.parseDouble(fields[i]);
                int[] idx = (isSequential) ?
                        new int[]{0, 0, i} :
                        new int[]{0, i};

                array.putScalar(idx, d);
            }

            Inference.Request request = new Inference.Request(Nd4jBase64.base64String(array));
            Inference.Response.Classify response = skilClient.classify(inferenceEndpoint, request);

            System.out.format("Inference response: %s\n", response.toString());
            if (label != null) {
                System.out.format("  Label expected: %s\n", label);
            }
        }

        br.close();
    }

    public static void main(String[] args) throws Exception {
        ModelServerDirectInferenceExample m = new ModelServerDirectInferenceExample();
        JCommander.newBuilder()
                .addObject(m)
                .build()
                .parse(args);

        m.run();
    }
}

