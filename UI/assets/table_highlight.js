/**
 * table_highlight.js
 * 
 * This script ensures only one cell can be highlighted across all
 * tables in the driftNavi application. It uses a Dash clientside callback
 * to properly integrate with Dash's reactivity system.
 */

// Register a clientside callback that handles table highlights
if (window.dash_clientside === undefined) {
    window.dash_clientside = {};
}

window.dash_clientside.table_sync = {
    // This function runs whenever any table's active cell changes
    // It returns a tuple of [null, null, null] or [active_cell, null, null] etc.
    // depending on which table was clicked
    syncTableSelections: function(primary_active, secondary_active, main_active) {
        // Get the callback context to identify which input triggered the callback
        const triggered = window.dash_clientside.callback_context.triggered;
        
        if (triggered.length && triggered[0].value !== null) {
            const triggerId = triggered[0].prop_id.split('.')[0];
            
            // Prepare the return values - all nulls by default
            const returnValues = [null, null, null];
            
            // Only keep the active cell for the table that was clicked
            if (triggerId === 'table-primary-overview') {
                returnValues[0] = primary_active;
            } else if (triggerId === 'table-secondary-overview') {
                returnValues[1] = secondary_active;
            } else if (triggerId === 'table-overview') {
                returnValues[2] = main_active;
            }
            
            return returnValues;
        }
        
        // If this wasn't triggered by a user action, don't change anything
        return [primary_active, secondary_active, main_active];
    }
};
