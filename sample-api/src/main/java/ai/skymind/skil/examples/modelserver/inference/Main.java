package ai.skymind.skil.examples.modelserver.inference;

import java.util.Arrays;

import org.slf4j.LoggerFactory;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class Main {
    public static void main (String... args) throws Exception {
        Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.ERROR);

        final String usage = "usage: quickstart | inference | directInference | normalizeUciData";

        if (args.length > 0) {
            String sub = args[0];
            String[] subArgs = Arrays.copyOfRange(args, 1, args.length);

            if (sub.equalsIgnoreCase("quickstart")) {
                if (args.length < 2) {
                    System.out.println("quickstart: missing URL");
                    System.out.println("usage: quickstart http://host:port/endpoints/...");
                    System.exit(1);
                }

                String url = args[1];
                if (!url.endsWith("/classify")) {
                    url += "/classify";
                }

                // Fetch UCI data + normalize it
                System.out.println("fetching UCI data and normalizing it");
                NormalizeUciData.main(new String[]{});

                // Run direct inference
                System.out.println("running inference on test data at " + url);
                String[] inferenceArgs = Arrays.asList(
                        "--inference", url,
                        "--input", NormalizeUciData.DEFAULT_TEST_OUTPUT,
                        "--sequential").toArray(new String[]{});

                ModelServerDirectInferenceExample.main(inferenceArgs);
            } else if (sub.equalsIgnoreCase("inference")) {
                ModelServerInferenceExample.main(subArgs);
            } else if (sub.equalsIgnoreCase("directInference")) {
                ModelServerDirectInferenceExample.main(subArgs);
            } else if (sub.equalsIgnoreCase("normalizeUciData")) {
                NormalizeUciData.main(subArgs);
            } else {
                System.out.println(String.format("unknown: %s", args[0]));
                System.out.println(usage);
                System.exit(1);
            }
        } else {
            System.out.println(usage);
            System.exit(1);
        }
    }
}
