package com.euromillioner;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.ResponseHandler;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Main {

    private static Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        try {
            String url = "http://portalseven.com/lottery/euromillions_winning_numbers.jsp?fromDate=1900-01-01&toDate=2020-06-14&viewType=3";

            HttpGet getRequest = new HttpGet(url);

            CloseableHttpClient httpClient = HttpClients.createDefault();

            ResponseHandler<String> responseHandler = response -> {
                int status = response.getStatusLine().getStatusCode();
                if (status >= 200 && status < 300) {
                    HttpEntity entity = response.getEntity();
                    return entity != null ? EntityUtils.toString(entity) : null;
                } else {
                    throw new ClientProtocolException("Unexpected response status: " + status);
                }
            };

            // Random sleep to avoid bot detection
            Thread.sleep((long) (Math.random() * 1000));

            String responseBody = httpClient.execute(getRequest, responseHandler);

            httpClient.close();

            Document document = Jsoup.parse(responseBody);

            Element resultsTable = document.getElementsByClass("table table-bordered table-condensed table-striped text-center table-hover").first();

            Elements elements = resultsTable.child(0).children();

            // Getting rid of the info row
            elements.remove(0);

            String csvLabel = "day_of_week, month, day, year, first, second, third, fourth, fift,; special_1, special_2,";

            File trainFile = File.createTempFile("emn",".csv");
            if (!trainFile.exists()) trainFile.createNewFile();

            FileWriter trainFileWriter = new FileWriter(trainFile,true);
            trainFileWriter.write(csvLabel);

            File validationFile = File.createTempFile("emn_validation",".csv");
            if (!trainFile.exists()) trainFile.createNewFile();

            FileWriter validationFileWriter = new FileWriter(validationFile,true);
            validationFileWriter.write(csvLabel);

            int percentageForTraining = 70;
            int marginElementsIndex = Double.valueOf((percentageForTraining / 100.0) * elements.size()).intValue();

            for (int i = 0; i < elements.size(); i++) {
                Elements tds = elements.get(i).children();
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < tds.size(); j++) {
                    String text = tds.get(j).text();
                    if (j == 0) {
                        DateTimeFormatter f = DateTimeFormatter.ofPattern("E, MMM d, yyyy");
                        LocalDate localDate = LocalDate.from(f.parse(text));
                        int dayOfWeek = localDate.getDayOfWeek().getValue();
                        int month = localDate.getMonthValue();
                        int day = localDate.getDayOfMonth();
                        int year = localDate.getYear();
                        sb.append(dayOfWeek + ", " + month + ", " + day + ", " + year + ", ");
                    } else {
                        sb.append(text + ", ");
                    }
                }
                if (i < marginElementsIndex) trainFileWriter.write(sb.toString());
                else if (i >= marginElementsIndex) validationFileWriter.write(sb.toString());
            }

            trainFileWriter.close();
            validationFileWriter.close();

            DMatrix trainMatrix = new DMatrix(trainFile.getAbsolutePath() + "?format=csv&label_column=0");
            DMatrix validationMatrix = new DMatrix(validationFile.getAbsolutePath()  + "?format=csv&label_column=0");

            Map<String, Object> params = new HashMap<String, Object>() {
                {
                    put("booster", "gbtree");
                    put("eta", 1.0);
                    put("max_depth", 3);
                    put("predictor", "cpu_predictor");
                    put("objective", "reg:logistic");
                    put("subsample", 1);
                    put("silent", 1);
                    put("nthread", 6);
                    put("gamma", 1.0);
                    put("eval_metric", "logloss");
                }
            };

            // Specify a watch list to see model accuracy on data sets
            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainMatrix);
                    put("test", validationMatrix);
                }
            };

            int nround = 500;
            Booster booster = XGBoost.train(trainMatrix, params, nround, watches, null, null);
            Booster boosterTest = XGBoost.train(validationMatrix, params, nround, watches, null, null);

            float[][] predict = booster.predict(trainMatrix);
            float[][] predict1 = boosterTest.predict(validationMatrix);

            System.out.println(checkPredicts(predict, predict1));
        } catch (IOException | InterruptedException | XGBoostError e) {
            logger.error("Could not access URL - " + e.getMessage());
            logger.debug(ExceptionUtils.getStackTrace(e));
        }
    }

    public static boolean checkPredicts(float[][] fPredicts, float[][] sPredicts) {
        if (fPredicts.length != sPredicts.length) {
            return false;
        }

        for (int i = 0; i < fPredicts.length; i++) {
            if (!Arrays.equals(fPredicts[i], sPredicts[i])) {
                return false;
            }
        }

        return true;
    }
}
