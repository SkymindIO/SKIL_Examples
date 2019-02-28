package ai.skymind.skil.examples.modelserver.inference.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Michael on 6/2/17.
 */
public class TransformedArray {
    public static class Request {
        public Request(String[] values) {
            this.values = values;
        }

        @JsonProperty("values")
        private String[] values;

        public String[] getValues() {
            return this.values;
        }
    }

    public static class BatchedRequest {
        public BatchedRequest(String[] values) {
            this(Arrays.asList(new Request(values)));
        }

        public BatchedRequest(List<Request> records) {
            this.records = records;
        }

        @JsonProperty("records")
        private List<Request> records;

        public List<Request> getRecords() {
            return this.records;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Response {
        @JsonProperty("ndarray")
        private String ndarray = null;

        public String getNdArray() {
            return this.ndarray;
        }
        public String toString() {
            return "ndarray{" + this.ndarray + "}";
        }
    }
}
