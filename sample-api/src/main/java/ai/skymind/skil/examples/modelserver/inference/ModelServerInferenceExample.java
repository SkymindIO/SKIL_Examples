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

        // Initialize RestTemplate
        RestTemplate restTemplate = new RestTemplate();

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

            final HttpHeaders requestHeaders = new HttpHeaders();
            final Object transformRequest;

            if (isSequential == true) {
                requestHeaders.add("Sequence", "true");
                transformRequest = new TransformedArray.BatchedRequest(fields);
            } else {
                transformRequest = new TransformedArray.Request(fields);
            }

            if (textAsJson) {
                // Accept JSON
                requestHeaders.setAccept(Arrays.asList(MediaType.APPLICATION_JSON));

                // Temp fix
                List<HttpMessageConverter<?>> converters = restTemplate.getMessageConverters();
                converters.add(new ExtendedMappingJackson2HttpMessageConverter());
                restTemplate.setMessageConverters(converters);
            }

            final HttpEntity<Object> httpEntity =
                    new HttpEntity<Object>(transformRequest, requestHeaders);

            final TransformedArray.Response arrayResponse = restTemplate.postForObject(
                    transformedArrayEndpoint,
                    httpEntity,
                    TransformedArray.Response.class);

            Class clazz;
            Object request;

            if (inferenceType == InferenceType.Single || inferenceType == InferenceType.Multi) {
                clazz = (inferenceType == InferenceType.Single) ?
                        Inference.Response.Classify.class : Inference.Response.MultiClassify.class;

                request = new Inference.Request(arrayResponse.getNdArray());

             } else {
                 clazz = Knn.Response.class;
                 request = new Knn.Request(knnN, arrayResponse.getNdArray());
             }

             final Object response = restTemplate.postForObject(
                         inferenceEndpoint,
                         request,
                         clazz);

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
