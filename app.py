import gradio as gr
from config import *
from event_handlers import *


# --- Gradio UI Definition ---
# theme = gr.themes.Default(primary_hue=gr.themes.colors.blue).set()
theme = gr.themes.Ocean(primary_hue=gr.themes.colors.blue).set()

with gr.Blocks(
    theme=theme,
    title="grAudio",
    css=".gradio-container { max-width: 95% !important; }",
) as app:

    # --- Global State ---
    initial_audio_list = load_existing_audio()
    audio_list_state = gr.State(value=initial_audio_list)
    newly_generated_state = gr.State([])
    # State to store the index of the selected row in the DataFrame
    selected_index_state = gr.State(-1)  # -1 means nothing selected

    # --- UI Layout ---
    gr.Markdown("# Generate Audio from text")
    with gr.Row(equal_height=False):
        # --- Column 1: Configuration (Left) ---
        with gr.Column(scale=2, min_width=350):
            gr.Markdown("### Generation Configuration")
            with gr.Accordion("Batch size & Temperatures", open=True):
                batch_size_number = gr.Number(
                    value=1,
                    label="Seed",
                    minimum=0,
                    step=1,
                    scale=1,
                )
                semantic_temp_slider = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.1, label="Semantic Temp"
                )
                coarse_temp_slider = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.1, label="Coarse Temp"
                )
                fine_temp_slider = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.1, label="Fine Temp"
                )
            with gr.Accordion("Model, Devices", open=True):
                model_type_dropdown = gr.Dropdown(
                    choices=["small", "large"], value="small", label="Model Type"
                )

                available_devices, best_device = get_available_torch_devices()
                device_dropdown = gr.Dropdown(
                    choices=available_devices, value=best_device, label="Device"
                )
            with gr.Accordion("Voice Prompt", open=True):
                prompt_dropdown = gr.Dropdown(
                    choices=get_available_prompts(),
                    label="Select Voice Prompt",
                    info="Optional",
                    multiselect=False,
                    allow_custom_value=False,
                )
                refresh_prompts_btn = gr.Button(
                    "Refresh Prompts", variant="secondary", size="sm"
                )
            with gr.Accordion("Create New Voice Prompt", open=False):
                prompt_audio_upload = gr.File(
                    value=None,
                    file_count="single",
                    label="Upload Audio (.wav, .mp3)",
                    file_types=["audio"],
                    type="filepath",
                )
                create_prompt_btn = gr.Button("Create Prompt", variant="secondary")

        # --- Column 2: Text Input & Generate Button (Middle) ---
        with gr.Column(scale=4, min_width=600):
            gr.Markdown("### Text Input")
            text_input_block = gr.Textbox(
                lines=30,
                placeholder="If your text includes multiple long sentences, select a voice prompt to have consistent speech.\nDo not use long sentence, split them out to multiple sentences with each less than 15 seconds",
                label="Text Prompts",
            )
            generate_btn = gr.Button("Generate", variant="primary")
        # --- Column 3: Generated Audio Display (Right) - SIMPLIFIED ---
        with gr.Column(scale=2, min_width=250):
            gr.Markdown("### Generated Audio")
            # DataFrame to display the list
            audio_dataframe = gr.DataFrame(
                headers=["File", "Prompt", "Duration (s)"],
                datatype=["str", "str", "str"],
                interactive=True,  # Allow row selection
                row_count=(10, "dynamic"),  # Show ~10 rows, scroll if more
                col_count=(3, "fixed"),
                # value=format_audio_list_for_dataframe(initial_audio_list) # Set initial value via app.load
            )
            # Single audio player for the selected item
            selected_audio_player = gr.Audio(
                label="Selected Audio",
                type="filepath",
                interactive=False,  # Only for playback
            )
            # Single delete button
            delete_selected_btn = gr.Button("Delete Selected Audio", variant="stop")

    # --- Event Handling ---

    # 1. Refresh Prompts Button
    refresh_prompts_btn.click(
        fn=update_available_prompts, inputs=None, outputs=[prompt_dropdown]
    )

    # 2. Create Prompt Button
    create_prompt_btn.click(
        fn=create_audio_prompt,
        inputs=[prompt_audio_upload, device_dropdown],
        outputs=[prompt_dropdown],
    )

    # 3. Generate Button -> Calls backend -> Outputs to temporary state
    generate_btn.click(
        fn=generate_batch_audio,
        inputs=[
            text_input_block,
            semantic_temp_slider,
            coarse_temp_slider,
            fine_temp_slider,
            batch_size_number,
            model_type_dropdown,
            device_dropdown,
            prompt_dropdown,
        ],
        outputs=[newly_generated_state],
    )

    # 4. Temporary State Change -> Updates the main audio list state
    newly_generated_state.change(
        fn=update_audio_list,
        inputs=[newly_generated_state, audio_list_state],
        outputs=[audio_list_state],
        show_progress="hidden",
    )

    # 5. Main Audio List State Change -> Updates the DataFrame display
    #    Also clears selection when the list updates.
    audio_list_state.change(
        fn=format_audio_list_for_dataframe,
        inputs=[audio_list_state],
        outputs=[audio_dataframe],
        show_progress="hidden",
    ).then(  # Chain: after updating dataframe, clear selection player and index
        fn=lambda: (None, -1),  # Function returning values to clear outputs
        inputs=None,
        outputs=[selected_audio_player, selected_index_state],
        show_progress="hidden",
        queue=False,
    )

    # 6. DataFrame Row Selection -> Updates the selected index and audio player
    audio_dataframe.select(
        fn=handle_row_selection,
        inputs=[audio_list_state],  # Pass the full list state to find the filepath
        outputs=[
            selected_audio_player,
            selected_index_state,
        ],
        show_progress="hidden",
    )

    # 7. Delete Selected Button Click -> Calls delete handler
    delete_selected_btn.click(
        fn=handle_delete_selected,
        inputs=[selected_index_state, audio_list_state],  # Pass index and list
        outputs=[
            audio_list_state,  # Update the main list state
            selected_index_state,  # Clear the selected index
            selected_audio_player,  # Clear the audio player
        ],
        show_progress="hidden",
    )

    # 8. Initial Load: Populate the DataFrame
    app.load(
        fn=format_audio_list_for_dataframe,
        inputs=[audio_list_state],  # Use the initial state value
        outputs=[audio_dataframe],  # Render initial data into the DataFrame
    )


if __name__ == "__main__":
    app.launch(debug=True, share=False)
