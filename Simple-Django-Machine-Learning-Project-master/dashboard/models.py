from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.svm import LinearSVC

# Create your models here.



class Data(models.Model):
    reviews = models.CharField(max_length=100, null=True)
    #Rate = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(19)], null=True)
    #height = models.PositiveIntegerField(null=True)
    #sex = models.PositiveIntegerField(choices=GENDER, null=True)
    predictions = models.CharField(max_length=100, blank=True)
    date = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        ml_model = joblib.load('ml_model/ml_reviews_model.joblib')
        v = joblib.load('ml_model/vectorizer.pkl')
        news_reviews2=""
        v1=v.transform([self.reviews])
        new_reviews=ml_model.predict(v1)
        if (new_reviews[0]==-1.0) :
            news_reviews2=news_reviews2+"Negative"
        else:
            news_reviews2=news_reviews2+"Positive"
        self.predictions = news_reviews2
        return super().save(*args, *kwargs)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return self.name
