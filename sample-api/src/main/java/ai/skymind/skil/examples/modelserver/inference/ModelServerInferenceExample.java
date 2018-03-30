package ai.skymind.skil.examples.modelserver.inference;

import ai.skymind.skil.examples.modelserver.inference.model.Knn;
import ai.skymind.skil.examples.modelserver.inference.model.Inference;
import ai.skymind.skil.examples.modelserver.inference.model.TransformedArray;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

public class ModelServerInferenceExample {

    private enum InferenceType {
        Multi,
        Single,
        Knn
    }

    @Parameter(names="--transform", description="Endpoint for Transform", required=true)
    private String transformedArrayEndpoint;

    @Parameter(names="--inference", description="Endpoint for Inference", required=true)
    private String inferenceEndpoint;

    @Parameter(names="--type", description="Type of endpoint (multi or single)", required=true)
    private InferenceType inferenceType;

    @Parameter(names="--input", description="CSV input file", required=true)
    private String inputFile;

    @Parameter(names="--sequential", description="If this transform a sequential one", required=false)
    private boolean isSequential = false;

    @Parameter(names="--knn", description="Number of K Nearest Neighbors to return", required=false)
    private int knnN = 20;

    @Parameter(names="--textAsJson", description="Parse text/plain as JSON", required=false, arity=1)
    private boolean textAsJson;

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
            String[] fields = line.split(",");

            // Maybe strip quotes
            for (int i=0; i<fields.length; i++) {
                final String field = fields[i];
                if (field.matches("^\".*\"$")) {
                    fields[i] = field.substring(1, field.length()-1);
                }
            }

            final TransformedArray.Response arrayResponse;
            if (isSequential) {
                arrayResponse = skilClient.transform(
                        transformedArrayEndpoint,
                        new TransformedArray.BatchedRequest(fields)
                );
            } else {
                arrayResponse = skilClient.transform(
                        transformedArrayEndpoint,
                        new TransformedArray.Request(fields)
                );
            }

            Object response;

            if (inferenceType == InferenceType.Single) {
                response = skilClient.classify(inferenceEndpoint, new Inference.Request(arrayResponse.getNdArray()));
            } else if (inferenceType == InferenceType.Multi) {
                response = skilClient.multiClassify(inferenceEndpoint, new Inference.Request(arrayResponse.getNdArray()));
            } else {
                response = skilClient.knn(inferenceEndpoint, new Knn.Request(knnN, arrayResponse.getNdArray()));
            }

             System.out.format("Inference response: %s\n", response.toString());
        }

        br.close();
    }

    public static void main(String[] args) throws Exception {
        ModelServerInferenceExample m = new ModelServerInferenceExample();
        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);

        m.run();
    }
}
