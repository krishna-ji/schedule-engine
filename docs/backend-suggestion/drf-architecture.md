# Django REST Framework Backend Architecture

## Your Situation: DRF Expert, Need Dependency Isolation

**Challenge**: Django + DEAP may need different Python versions or have conflicting dependencies (numpy, matplotlib versions).

**Solution**: DRF for API, isolated Python worker for GA engine via Celery.

---

## Architecture: DRF + Celery + Redis

```
┌─────────────────────┐      ┌──────────────┐      ┌──────────────────────┐
│  Django REST API    │─────▶│    Redis     │─────▶│  Celery Worker       │
│  (Views, Models)    │      │   (Broker)   │      │  (GA Engine Runner)  │
└─────────────────────┘      └──────────────┘      └──────────────────────┘
         │                                                     │
         ▼                                                     ▼
   PostgreSQL                                         S3/FileSystem
   (Users, Jobs, Metadata)                           (Schedule Outputs)
```

**Key Principle**: Django runs in one Python environment, Celery worker runs in another (isolated via Docker or virtualenv).

---

## Stack Components

1. **Django 4.2+** – Main framework (Python 3.10-3.12)
2. **Django REST Framework** – API layer you already know
3. **Celery 5.x** – Async task queue for long-running GA jobs
4. **Redis** – Message broker + result backend
5. **PostgreSQL** – Primary database
6. **Docker** – Isolate GA worker environment (Python 3.9 if needed)

---

## Project Structure

```
schedule-saas/
├── manage.py
├── requirements.txt                    # Django/DRF dependencies
├── requirements-worker.txt             # DEAP + GA engine dependencies
│
├── config/                             # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── celery.py                       # Celery configuration
│
├── apps/
│   ├── accounts/                       # User management
│   │   ├── models.py
│   │   ├── serializers.py
│   │   └── views.py
│   │
│   └── schedules/                      # Core scheduling app
│       ├── models.py                   # ScheduleJob, ScheduleResult models
│       ├── serializers.py              # Input/output validation
│       ├── views.py                    # API endpoints
│       ├── tasks.py                    # Celery tasks (calls GA engine)
│       └── urls.py
│
├── ga_engine/                          # Your existing schedule-engine code
│   ├── src/
│   ├── config/
│   └── main.py
│
├── docker/
│   ├── api.Dockerfile                  # Django API container
│   └── worker.Dockerfile               # Celery worker container
│
└── docker-compose.yml
```

---

## Step-by-Step Implementation

### Phase 1: Django Project Setup

#### 1.1 Install Django Dependencies
```bash
pip install django djangorestframework celery redis psycopg2-binary django-cors-headers djangorestframework-simplejwt
```

#### 1.2 Create Django Models
```python
# apps/schedules/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid

class ScheduleJob(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='schedule_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Input data (stored as JSON)
    courses = models.JSONField()
    groups = models.JSONField()
    instructors = models.JSONField()
    rooms = models.JSONField()
    config = models.JSONField(default=dict)  # pop_size, generations, etc.
    
    # Results
    result_data = models.JSONField(null=True, blank=True)
    output_files = models.JSONField(default=list)  # URLs to S3/files
    error_message = models.TextField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # GA metrics
    hard_violations = models.IntegerField(null=True, blank=True)
    soft_penalty = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"ScheduleJob {self.id} - {self.status}"
```

#### 1.3 Create DRF Serializers
```python
# apps/schedules/serializers.py
from rest_framework import serializers
from .models import ScheduleJob

class ScheduleInputSerializer(serializers.Serializer):
    """Validates user input before creating job."""
    courses = serializers.ListField(child=serializers.DictField())
    groups = serializers.ListField(child=serializers.DictField())
    instructors = serializers.ListField(child=serializers.DictField())
    rooms = serializers.ListField(child=serializers.DictField())
    config = serializers.DictField(required=False, default=dict)
    
    def validate_config(self, value):
        """Validate GA parameters."""
        pop_size = value.get('pop_size', 100)
        generations = value.get('generations', 500)
        
        if pop_size < 20 or pop_size > 500:
            raise serializers.ValidationError("pop_size must be between 20 and 500")
        if generations < 10 or generations > 2000:
            raise serializers.ValidationError("generations must be between 10 and 2000")
        
        return value

class ScheduleJobSerializer(serializers.ModelSerializer):
    """Returns job status and results."""
    class Meta:
        model = ScheduleJob
        fields = [
            'id', 'status', 'created_at', 'started_at', 'completed_at',
            'hard_violations', 'soft_penalty', 'output_files', 'error_message'
        ]
        read_only_fields = fields

class ScheduleResultSerializer(serializers.ModelSerializer):
    """Returns full results including schedule data."""
    class Meta:
        model = ScheduleJob
        fields = [
            'id', 'status', 'result_data', 'output_files',
            'hard_violations', 'soft_penalty', 'completed_at'
        ]
        read_only_fields = fields
```

#### 1.4 Create API Views
```python
# apps/schedules/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import ScheduleJob
from .serializers import (
    ScheduleInputSerializer,
    ScheduleJobSerializer,
    ScheduleResultSerializer
)
from .tasks import run_ga_schedule_task

class ScheduleViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoints for schedule generation.
    
    POST /api/schedules/          - Create new schedule job
    GET  /api/schedules/          - List user's jobs
    GET  /api/schedules/{id}/     - Get job status
    GET  /api/schedules/{id}/result/ - Get full results
    """
    permission_classes = [IsAuthenticated]
    serializer_class = ScheduleJobSerializer
    
    def get_queryset(self):
        """Users see only their own jobs."""
        return ScheduleJob.objects.filter(user=self.request.user)
    
    @action(detail=False, methods=['post'])
    def create_schedule(self, request):
        """
        Create new schedule generation job.
        
        Request body:
        {
            "courses": [...],
            "groups": [...],
            "instructors": [...],
            "rooms": [...],
            "config": {"pop_size": 100, "generations": 500}
        }
        """
        serializer = ScheduleInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Create job record
        job = ScheduleJob.objects.create(
            user=request.user,
            courses=serializer.validated_data['courses'],
            groups=serializer.validated_data['groups'],
            instructors=serializer.validated_data['instructors'],
            rooms=serializer.validated_data['rooms'],
            config=serializer.validated_data.get('config', {}),
        )
        
        # Queue Celery task (async)
        run_ga_schedule_task.delay(str(job.id))
        
        return Response(
            {
                'job_id': str(job.id),
                'status': 'PENDING',
                'message': 'Schedule generation started'
            },
            status=status.HTTP_202_ACCEPTED
        )
    
    @action(detail=True, methods=['get'])
    def result(self, request, pk=None):
        """Get full schedule results (only if completed)."""
        job = get_object_or_404(self.get_queryset(), pk=pk)
        
        if job.status != 'SUCCESS':
            return Response(
                {'error': 'Job not completed yet'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = ScheduleResultSerializer(job)
        return Response(serializer.data)
```

#### 1.5 Setup URLs
```python
# apps/schedules/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ScheduleViewSet

router = DefaultRouter()
router.register(r'schedules', ScheduleViewSet, basename='schedule')

urlpatterns = [
    path('', include(router.urls)),
]

# config/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('apps.schedules.urls')),
    path('api/auth/', include('rest_framework.urls')),  # DRF browsable API login
]
```

---

### Phase 2: Celery Worker Setup

#### 2.1 Configure Celery
```python
# config/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('schedule_saas')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# config/__init__.py
from .celery import app as celery_app
__all__ = ('celery_app',)
```

#### 2.2 Add Celery Settings
```python
# config/settings.py (add these)
# Celery Configuration
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes max per job
```

#### 2.3 Create Celery Task (Bridge to GA Engine)
```python
# apps/schedules/tasks.py
import json
import tempfile
from pathlib import Path
from celery import shared_task
from django.utils import timezone
from .models import ScheduleJob

# Import your GA engine
import sys
sys.path.append('/app/ga_engine')  # Adjust path based on deployment
from src.workflows import run_standard_workflow

@shared_task(bind=True)
def run_ga_schedule_task(self, job_id: str):
    """
    Execute GA scheduling in background worker.
    
    This runs in isolated Python environment with DEAP installed.
    """
    try:
        # Get job from database
        job = ScheduleJob.objects.get(id=job_id)
        job.status = 'RUNNING'
        job.started_at = timezone.now()
        job.save()
        
        # Create temporary directory for GA engine I/O
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare input files (GA engine expects JSON files)
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            (data_dir / "Course.json").write_text(json.dumps(job.courses))
            (data_dir / "Groups.json").write_text(json.dumps(job.groups))
            (data_dir / "Instructors.json").write_text(json.dumps(job.instructors))
            (data_dir / "Rooms.json").write_text(json.dumps(job.rooms))
            
            # Run GA engine (your existing code)
            result = run_standard_workflow(
                pop_size=job.config.get('pop_size', 100),
                generations=job.config.get('generations', 500),
                crossover_prob=job.config.get('crossover_prob', 0.7),
                mutation_prob=job.config.get('mutation_prob', 0.2),
                data_dir=str(data_dir),
                output_dir=str(Path(tmpdir) / "output"),
                validate=True,
            )
            
            # Read output files
            output_path = Path(result["output_path"])
            schedule_json = (output_path / "schedule.json").read_text()
            
            # TODO: Upload plots to S3/MinIO (for now, store locally)
            output_files = []
            for png_file in output_path.glob("*.png"):
                # Copy to media directory or upload to S3
                # output_files.append(upload_to_s3(png_file))
                output_files.append(str(png_file.name))
            
            # Update job with results
            job.status = 'SUCCESS'
            job.result_data = json.loads(schedule_json)
            job.output_files = output_files
            job.hard_violations = int(result["best_individual"].fitness.values[0])
            job.soft_penalty = float(result["best_individual"].fitness.values[1])
            job.completed_at = timezone.now()
            job.save()
            
    except Exception as e:
        # Handle errors
        job.status = 'FAILED'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save()
        raise  # Re-raise for Celery to log
```

---

### Phase 3: Dependency Isolation with Docker

#### 3.1 API Dockerfile (Django + DRF)
```dockerfile
# docker/api.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Django dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Django project
COPY manage.py .
COPY config/ ./config/
COPY apps/ ./apps/

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD python manage.py migrate && \
    python manage.py runserver 0.0.0.0:8000
```

**requirements.txt** (Django/DRF only):
```
Django>=4.2,<5.0
djangorestframework>=3.14
celery>=5.3
redis>=5.0
psycopg2-binary>=2.9
django-cors-headers>=4.0
djangorestframework-simplejwt>=5.2
```

#### 3.2 Worker Dockerfile (Celery + GA Engine)
```dockerfile
# docker/worker.Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install GA engine dependencies
COPY requirements-worker.txt .
RUN pip install --no-cache-dir -r requirements-worker.txt

# Copy Django models (needed for Celery tasks)
COPY manage.py .
COPY config/ ./config/
COPY apps/ ./apps/

# Copy GA engine
COPY ga_engine/ ./ga_engine/

# Run Celery worker
CMD celery -A config worker --loglevel=info
```

**requirements-worker.txt** (DEAP + GA dependencies):
```
Django>=4.2,<5.0
celery>=5.3
redis>=5.0
psycopg2-binary>=2.9

# GA Engine dependencies
deap>=1.4.1
tqdm>=4.66.0
matplotlib>=3.7.0
numpy>=1.24.0
```

#### 3.3 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: schedule_db
      POSTGRES_USER: schedule_user
      POSTGRES_PASSWORD: schedule_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://schedule_user:schedule_pass@db:5432/schedule_db
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  worker:
    build:
      context: .
      dockerfile: docker/worker.Dockerfile
    command: celery -A config worker --loglevel=info
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgresql://schedule_user:schedule_pass@db:5432/schedule_db
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

---

## Running the Stack

### Development (Local)

```bash
# Start all services
docker-compose up -d

# Create Django superuser
docker-compose exec api python manage.py createsuperuser

# Check worker logs
docker-compose logs -f worker

# Stop all services
docker-compose down
```

### Without Docker (Using Virtualenvs)

```bash
# Terminal 1: Django API
cd schedule-saas
python -m venv venv-api
venv-api\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Terminal 2: Celery Worker (different virtualenv!)
python -m venv venv-worker
venv-worker\Scripts\activate
pip install -r requirements-worker.txt
celery -A config worker --loglevel=info

# Terminal 3: Redis
# Install Redis for Windows from: https://github.com/microsoftarchive/redis/releases
redis-server
```

---

## Testing the API

### 1. Create User & Get Token
```bash
# Using Django admin or DRF browsable API
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Response: {"token": "abc123..."}
```

### 2. Submit Schedule Job
```bash
curl -X POST http://localhost:8000/api/schedules/create_schedule/ \
  -H "Authorization: Token abc123..." \
  -H "Content-Type: application/json" \
  -d @sample_input.json

# Response:
# {
#   "job_id": "uuid-1234",
#   "status": "PENDING",
#   "message": "Schedule generation started"
# }
```

### 3. Check Job Status
```bash
curl http://localhost:8000/api/schedules/uuid-1234/ \
  -H "Authorization: Token abc123..."

# Response:
# {
#   "id": "uuid-1234",
#   "status": "RUNNING",  # or SUCCESS, FAILED
#   "created_at": "2025-10-16T10:30:00Z",
#   "hard_violations": null,
#   "soft_penalty": null
# }
```

### 4. Get Results
```bash
curl http://localhost:8000/api/schedules/uuid-1234/result/ \
  -H "Authorization: Token abc123..."

# Response:
# {
#   "id": "uuid-1234",
#   "status": "SUCCESS",
#   "result_data": {...full schedule JSON...},
#   "output_files": ["evolution_plot.png", "calendar.png"],
#   "hard_violations": 0,
#   "soft_penalty": 12.5
# }
```

---

## Handling Python Version Conflicts

### Scenario: Django needs Python 3.11, DEAP needs Python 3.9

**Solution 1: Docker (Recommended)**
- API container uses `python:3.11-slim`
- Worker container uses `python:3.9-slim`
- Completely isolated, no conflicts

**Solution 2: Separate Virtual Environments**
```
/schedule-saas
  /venv-api       ← Python 3.11 (Django)
  /venv-worker    ← Python 3.9 (DEAP)
```
Run worker with: `venv-worker\Scripts\python -m celery -A config worker`

**Solution 3: Pyenv (Mac/Linux)**
```bash
pyenv install 3.11.0
pyenv install 3.9.0

cd schedule-saas
pyenv local 3.11.0  # For Django

cd worker/
pyenv local 3.9.0   # For Celery worker
```

---

## Adding Real-Time Progress Updates

### Option A: WebSockets (Django Channels)
```python
# Install: pip install channels channels-redis

# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ScheduleProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.job_id = self.scope['url_route']['kwargs']['job_id']
        await self.channel_layer.group_add(
            f'schedule_{self.job_id}',
            self.channel_name
        )
        await self.accept()
    
    async def schedule_progress(self, event):
        await self.send(text_data=json.dumps({
            'progress': event['progress'],
            'generation': event['generation']
        }))

# tasks.py (emit progress from Celery)
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

@shared_task
def run_ga_schedule_task(job_id: str):
    channel_layer = get_channel_layer()
    
    # Inside GA loop (modify your code slightly)
    for gen in range(NGEN):
        # ... GA iteration ...
        
        # Emit progress
        async_to_sync(channel_layer.group_send)(
            f'schedule_{job_id}',
            {
                'type': 'schedule_progress',
                'progress': int((gen / NGEN) * 100),
                'generation': gen
            }
        )
```

### Option B: Polling (Simpler)
```python
# Add progress field to model
class ScheduleJob(models.Model):
    progress = models.IntegerField(default=0)  # 0-100

# Frontend polls every 2 seconds
GET /api/schedules/{job_id}/ → {"progress": 45, "status": "RUNNING"}
```

---

## Deployment Checklist

### Production Settings
```python
# config/settings.py
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com']

# Security
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Celery
CELERY_TASK_ALWAYS_EAGER = False  # Ensure tasks run async
CELERY_BROKER_URL = os.environ['REDIS_URL']

# Static files
STATIC_ROOT = '/app/staticfiles'
```

### Docker Production
```yaml
# docker-compose.prod.yml
services:
  api:
    image: your-registry/schedule-api:latest
    command: gunicorn config.wsgi:application --bind 0.0.0.0:8000
    
  worker:
    image: your-registry/schedule-worker:latest
    command: celery -A config worker --concurrency=4
```

### Scaling Workers
```bash
# Run 5 worker instances
docker-compose up --scale worker=5
```

---

## Advantages of This Approach

✅ **Use DRF**: Leverage your existing Django knowledge  
✅ **Isolated Dependencies**: Docker containers prevent conflicts  
✅ **Scalable**: Add workers as load increases  
✅ **Async Processing**: API responds instantly, jobs run in background  
✅ **Standard Django Patterns**: Models, serializers, views you know  
✅ **Battle-Tested**: Django + Celery used by Instagram, Mozilla, etc.  

---

## Next Steps

1. **Create Django Project**
   ```bash
   django-admin startproject config .
   python manage.py startapp schedules
   ```

2. **Copy GA Engine**
   ```bash
   cp -r ../schedule-engine/src ./ga_engine/src
   cp -r ../schedule-engine/config ./ga_engine/config
   ```

3. **Setup Models & Views** (use code above)

4. **Test Locally**
   ```bash
   docker-compose up
   ```

5. **Add Auth** (JWT or Django sessions)

6. **Deploy** (AWS ECS, DigitalOcean, Railway, etc.)

---

## Resources

- **DRF Tutorial**: https://www.django-rest-framework.org/tutorial/quickstart/
- **Celery + Django**: https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html
- **Docker Compose**: https://docs.docker.com/compose/
- **Django Channels** (WebSockets): https://channels.readthedocs.io/

---

**Summary**: Use DRF for API you're comfortable with, run Celery worker in separate Docker container with DEAP dependencies. Zero dependency conflicts, fully scalable, production-ready.
