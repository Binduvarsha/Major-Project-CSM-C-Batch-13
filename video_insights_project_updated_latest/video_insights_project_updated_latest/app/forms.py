from django import forms

class VideoForm(forms.Form):
    title = forms.CharField(
        max_length=255,
        label="Video Title",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control','rows': 3}),
        label="Video Description"
    )
    url = forms.URLField(
        required=False,
        label="Video URL",
        widget=forms.URLInput(attrs={'class': 'form-control'})
    )
    video_file = forms.FileField(
        required=False,
        label="Upload Video File",
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )

    def clean(self):
        cleaned_data = super().clean()
        url = cleaned_data.get("url")
        video_file = cleaned_data.get("video_file")

        # Ensure only one of the fields is provided
        if not url and not video_file:
            raise forms.ValidationError("You must provide either a URL or upload a video file.")
        if url and video_file:
            raise forms.ValidationError("You can only provide one input: either a URL or a video file.")
        return cleaned_data
