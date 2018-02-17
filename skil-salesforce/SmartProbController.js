({
	myAction : function(component, event, helper) {
		
	},
    
    doInit: function(component, event, helper) {
        var action = component.get("c.predict");
        action.setParams({recordId: component.get("v.recordId")});
        action.setCallback(this, function(response) {
            var prob = response.getReturnValue();
            component.set("v.smartProb", prob);
        });
        
        $A.enqueueAction(action);
    }
})