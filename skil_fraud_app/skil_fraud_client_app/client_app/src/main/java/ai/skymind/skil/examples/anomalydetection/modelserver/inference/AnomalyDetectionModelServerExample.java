package ai.skymind.skil.examples.anomalydetection.modelserver.inference;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.json.JSONObject;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

import ai.skymind.skil.examples.anomalydetection.modelserver.auth.Authorization;

public class AnomalyDetectionModelServerExample {


    @Parameter(names="--endpoint", description="Endpoint for classification", required=true)
    private String skilInferenceEndpoint = "";

    @Parameter(names="--feature", description="CSV feature file", required=true)
    private String inputCSVFeatureFile = "";

    public void run() throws Exception, IOException {

        File CSVFeatureFile = null;

        INDArray finalRecord = null;

        if ("blank".equals(inputCSVFeatureFile)) {

            finalRecord = Nd4j.zeros( 1, 10);

            System.out.println( "Generating blank test csv ..." );

        } else {

            CSVFeatureFile = new File( inputCSVFeatureFile );

            if (!CSVFeatureFile.exists() || !CSVFeatureFile.isFile()) {
                System.err.format("unable to access file %s\n", inputCSVFeatureFile);
                System.exit(2);
            } else {
                System.out.println( "Inference for: " + inputCSVFeatureFile );
            }

            CSVRecordReader rr1 = new CSVRecordReader(1);
            rr1.initialize(new FileSplit(CSVFeatureFile));

            DataSetIterator dsi = new RecordReaderDataSetIterator(rr1, 1);

            finalRecord = dsi.next().getFeatures();

        }

        String CSVBase64 = Nd4jBase64.base64String(finalRecord);
        System.out.println( CSVBase64 );
        System.out.println( "Finished CSV conversion" );
        skilClientGetInference( CSVBase64 );
    }

    private void skilClientGetInference( String CSVBase64 ) {

        Authorization auth = new Authorization();

        String auth_token = auth.getAuthToken( "admin", "admin" );

        System.out.println( "auth token: " + auth_token );

        try {

            String returnVal =
                    Unirest.post( skilInferenceEndpoint + "classify" )
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .header( "Authorization", "Bearer " + auth_token)
                            .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                    .put( "id", "some_id" )
                                    .put("prediction", new JSONObject().put("array", CSVBase64))
                                    .toString())
                            .asJson()
                            .getBody().getObject().toString();

            System.out.println( "classification return: " + returnVal );

        } catch (UnirestException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        AnomalyDetectionModelServerExample m = new AnomalyDetectionModelServerExample();

        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);

        m.run();
    }
}
