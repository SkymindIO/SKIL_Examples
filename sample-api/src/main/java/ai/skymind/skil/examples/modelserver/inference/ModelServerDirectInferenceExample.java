package ai.skymind.skil.examples.modelserver.inference;

import ai.skymind.skil.examples.modelserver.inference.model.Inference;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;

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

        // Initialize RestTemplate
        RestTemplate restTemplate = new RestTemplate();

        if (textAsJson) {
            List<HttpMessageConverter<?>> converters = restTemplate.getMessageConverters();
            converters.add(new ExtendedMappingJackson2HttpMessageConverter());
            restTemplate.setMessageConverters(converters);
        }

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
            final Object response = restTemplate.postForObject(
                    inferenceEndpoint,
                    request,
                    Inference.Response.Classify.class);

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

