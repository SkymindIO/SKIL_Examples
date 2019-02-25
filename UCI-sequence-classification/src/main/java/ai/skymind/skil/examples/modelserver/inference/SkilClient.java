package ai.skymind.skil.examples.modelserver.inference;

import ai.skymind.skil.examples.modelserver.inference.model.Inference;
import ai.skymind.skil.examples.modelserver.inference.model.Knn;
import ai.skymind.skil.examples.modelserver.inference.model.Login;
import ai.skymind.skil.examples.modelserver.inference.model.TransformedArray;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.web.client.RestTemplate;

import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class SkilClient {
    private RestTemplate restTemplate = new RestTemplate();
    private String token = null;

    public SkilClient(boolean textAsJson) {
        // Initialize RestTemplate

        if (textAsJson) {
            List<HttpMessageConverter<?>> converters = restTemplate.getMessageConverters();
            converters.add(new ExtendedMappingJackson2HttpMessageConverter());
            restTemplate.setMessageConverters(converters);
        }
    }

    private void login(String endpoint) {
        if (token == null) {
            URI endpointUri = URI.create(endpoint);
            String scheme = endpointUri.getScheme();
            String host = endpointUri.getHost();
            int port = endpointUri.getPort();

            String loginUrl = scheme + "://" + host + ":" + port + "/login";

            try {
                System.out.println("Logging into SKIL with userId 'admin'");
                ResponseEntity<Login.Response> res = restTemplate.postForEntity(
                        loginUrl,
                        new Login.Request("admin", "admin"),
                        Login.Response.class
                );
                token = res.getBody().token;
            } catch (Exception e) {
                System.err.println("Login failed: " + e.getMessage());
                e.printStackTrace();
            }

            if (token == null) {
                System.err.println("Failed to login to SKIL.");
                System.exit(3);
            }
        }
    }

    private <T> HttpEntity<T> createEntity(T entity) {
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + token);
        headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

        return new HttpEntity<>(entity, headers);
    }

    public Inference.Response.Classify classify(String endpoint, Inference.Request request) {
        login(endpoint);

        return restTemplate.postForObject(
                endpoint,
                createEntity(request),
                Inference.Response.Classify.class
        );
    }

    public Inference.Response.MultiClassify multiClassify(String endpoint, Inference.Request request) {
        login(endpoint);

        return restTemplate.postForObject(
                endpoint,
                createEntity(request),
                Inference.Response.MultiClassify.class
        );    }

    public Knn.Response knn(String endpoint, Knn.Request request) {
        login(endpoint);

        return restTemplate.postForObject(
                endpoint,
                createEntity(request),
                Knn.Response.class
        );
    }

    public TransformedArray.Response transform(String endpoint, TransformedArray.Request request) {
        login(endpoint);

        return restTemplate.postForObject(
                endpoint,
                createEntity(request),
                TransformedArray.Response.class
        );
    }

    public TransformedArray.Response transform(String endpoint, TransformedArray.BatchedRequest request) {
        login(endpoint);

        HttpEntity<TransformedArray.BatchedRequest> entity = createEntity(request);
        entity.getHeaders().add("Sequence", "true");

        return restTemplate.postForObject(
                endpoint,
                entity,
                TransformedArray.Response.class
        );
    }
}
