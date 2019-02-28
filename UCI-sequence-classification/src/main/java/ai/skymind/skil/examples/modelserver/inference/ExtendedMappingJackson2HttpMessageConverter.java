package ai.skymind.skil.examples.modelserver.inference;

import org.springframework.http.MediaType;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;

import java.util.ArrayList;
import java.util.List;

class ExtendedMappingJackson2HttpMessageConverter extends MappingJackson2HttpMessageConverter {
    public ExtendedMappingJackson2HttpMessageConverter() {
        List<MediaType> types = new ArrayList<MediaType>(super.getSupportedMediaTypes());
        types.add(new MediaType("text", "plain", DEFAULT_CHARSET));
        super.setSupportedMediaTypes(types);
    }
}