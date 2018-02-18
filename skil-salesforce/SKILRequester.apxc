public class SKILRequester {
    private class LoginResponse { public String token; }
    private class TransformResponse { public String ndarray; }
    private class PredictResponse { public Integer[] results; public Decimal[] probabilities; }
    
    private static String SKIL_HOST = '<skil_host_address>:9008';
    
    @AuraEnabled
    public static String predict(Id recordId) {
        Opportunity opp = [ SELECT Amount, Probability, ExpectedRevenue, Type, LeadSource, FiscalQuarter FROM Opportunity WHERE Id = :recordId ];
        String auth = login();
        String ndarray = transform(
            auth,
            opp.Amount,
            opp.Probability,
            opp.ExpectedRevenue, 
            opp.Type, 
            opp.LeadSource,
        	opp.FiscalQuarter);
        
        String pred = predict(auth, ndarray);
        
        return pred;
    }
    
    private static String predict(String auth, String ndarray) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(SKIL_HOST + '/endpoints/demos/model/smartprob/default/classify');
        req.setMethod('POST');
        
        Blob generatedBlob = Crypto.GenerateAESKey(128);
        String hex = EncodingUtil.ConvertTohex(generatedBlob);
        String guid = hex.substring(0, 8)
            + '-' + hex.substring(8, 12)
            + '-' + hex.substring(12, 16)
            + '-' + hex.substring(16, 20)
            + '-' + hex.substring(20);

        String jsonReq = '{"id": "' + guid + '", "prediction": { "array": "' + ndarray + '" } }';
        
        req.setBody(jsonReq);
        
        req.setHeader('Authorization', 'Bearer ' + auth);
        req.setHeader('Content-Type', 'application/json');
     	
        Http http = new Http();
     	HTTPResponse res = http.send(req);
     	String jsonBody = res.getBody();
        PredictResponse prediction = (PredictResponse)JSON.deserialize(jsonBody, PredictResponse.class);
        String[] results = new String[] { '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%', '100%' };
       
        return results[prediction.results[0]];
    }
    
    private static String transform(String auth, Decimal amount, Decimal prob, Decimal expectedRevenue, String type, String lead, Integer quarter) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(SKIL_HOST + '/endpoints/demos/datavec/smartprobtransform/default/transformincrementalarray');
        req.setMethod('POST');
        
        if (type == null) {
            type = '';
        }
        
        if (lead == null) {
            lead = '';
        }
        
        String csv = String.format('"{0}", "{1}", "{2}", "{3}", "{4}", "{5}"', new String[] { String.valueOf(amount.intValue()), prob.toPlainString(), String.valueOf(expectedRevenue.intValue()), type, lead, String.valueOf(quarter) });
        String jsonReq = '{"values": [' + csv + ']}';
        
        req.setBody(jsonReq);
        
        req.setHeader('Authorization', 'Bearer ' + auth);
        req.setHeader('Content-Type', 'application/json');
     	
        Http http = new Http();
     	HTTPResponse res = http.send(req);
     	String jsonBody = res.getBody();
        TransformResponse transformRes = (TransformResponse)JSON.deserialize(jsonBody, TransformResponse.class);
        return transformRes.ndarray.replace('\r\n', '\\r\\n');
    }
     
    private static String login() {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(SKIL_HOST + '/login');
        req.setMethod('POST');
        
        String loginReq = '{"userId": "admin", "password": "admin"}';
        
        req.setBody(loginReq);
        
        //req.setHeader('Authorization', authorizationHeader);
        req.setHeader('Content-Type', 'application/json');
     	
        Http http = new Http();
     	HTTPResponse res = http.send(req);
     	String jsonBody = res.getBody();
        LoginResponse loginRes = (LoginResponse)JSON.deserialize(jsonBody, LoginResponse.class);
        return loginRes.token;
    }
}