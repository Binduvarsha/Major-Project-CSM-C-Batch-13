from django.db import models

# Create your models here.
class User(models.Model):
    full_name = models.CharField(max_length=100, verbose_name="User Name")
    email = models.EmailField(verbose_name="Email")
    password = models.CharField(max_length=128, verbose_name="Password")
    phone_number = models.CharField(max_length=15, verbose_name="Phone Number")
    age =models.CharField(max_length=15, verbose_name="age")
    address = models.TextField(verbose_name="Address")
    photo = models.ImageField(upload_to='profiles/', verbose_name="Upload Profile", null=True, blank=True)
    otp = models.CharField(max_length=6, default='000000', help_text='Enter OTP for verification')
    otp_status = models.CharField(max_length=15, default='Not Verified', help_text='OTP status')
    status = models.CharField(max_length=15,default='Pending')

    def __str__(self):
        return self.full_name
    




class Feedback(models.Model):
    
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    user_name = models.CharField(max_length=50)
    user_email = models.EmailField(max_length=50)
    rating = models.IntegerField()
    additional_comments = models.TextField()
    
    class Meta:
        db_table = 'Feedback_details'
    


class Video(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='videos',blank=True, null=True)
    title = models.CharField(max_length=255)  # Title of the video
    description = models.TextField()  # Description of the video
    url = models.URLField(blank=True, null=True)  # URL of the video (optional for file uploads)
    source = models.CharField(
        max_length=50,
        choices=[
            ('YouTube', 'YouTube'),
            ('Google Drive', 'Google Drive'),
            ('Dropbox', 'Dropbox'),
            ('Uploaded File', 'Uploaded File')
        ]  # Source of the video
    )
    transcript = models.TextField(blank=True, null=True)  # Full transcribed text
    file_name = models.CharField(max_length=255, blank=True, null=True)  # Store actual file name
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp when created

    def __str__(self):
        return self.title


class TranscriptSegment(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='segments')  # Link to Video
    start_time = models.FloatField()  # Start time in seconds
    end_time = models.FloatField()  # End time in seconds
    text = models.TextField()  # Transcribed text for this segment

    def __str__(self):
        return f"{self.text[:50]}... ({self.start_time} - {self.end_time}s)"
