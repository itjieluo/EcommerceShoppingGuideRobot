from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ShopingGuideSystem.service.CentralControl import CentralControl
import json

# Start Aiger engine
CC = CentralControl()
CC.Init()
# Create your views here.

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def Dialog_sys(request):
    if request.method == 'POST':
        query = request.POST.get('data')
        print(query)
        result = CC.Sess(query)
        print(result)
        result_type = result[0]
        result_centence = result[1]
        result_product = result[2]

        data = str(result_centence)
        data = data
    return JsonResponse({"data":data})
