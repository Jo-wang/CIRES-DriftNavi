// JavaScript handlers for import/export functionality
document.addEventListener('DOMContentLoaded', function() {
    // Listen for the custom events from our clientside callbacks
    document.addEventListener('open-import-modal', function(event) {
        // Extract the dataset type from the event
        const datasetType = event.detail.datasetType;
        
        // Trigger the modal opening via Dash callback
        // We need to update a hidden div that will trigger our callback chain
        // First, we need to update the dataset target store
        if (datasetType === 'primary') {
            // Simulate a click on the placeholder to trigger the callback
            const event = new Event('click');
            const placeholder = document.getElementById('import-primary-placeholder');
            if (placeholder) {
                placeholder.dispatchEvent(event);
            }
        } else if (datasetType === 'secondary') {
            // Simulate a click on the placeholder to trigger the callback
            const event = new Event('click');
            const placeholder = document.getElementById('import-secondary-placeholder');
            if (placeholder) {
                placeholder.dispatchEvent(event);
            }
        }
        
        // Now manually open the modal
        // This is a workaround since we don't have direct access to the Dash modal from JS
        // The alternative is to create a callback in Python that opens the modal
        const modalElement = document.getElementById('upload-modal');
        if (modalElement) {
            // Add classes that Bootstrap uses to show modals
            modalElement.classList.add('show');
            modalElement.style.display = 'block';
            
            // Add backdrop
            const backdrop = document.createElement('div');
            backdrop.classList.add('modal-backdrop', 'fade', 'show');
            document.body.appendChild(backdrop);
            
            // Set body class
            document.body.classList.add('modal-open');
        }
    });
});
