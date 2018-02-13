package ai.skymind.skil.examples.mnist.modelserver.inference.model;
//package ai.skymind.skil.examples.mnist.modelserver.inference.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;

public class TransformedImage {
    public static class Request {
        public Request(String url) {
            this.url = url;
        }

        @JsonProperty("URL")
        private String url;

        public String getUrl() {
            return this.url;
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