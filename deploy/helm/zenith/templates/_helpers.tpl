{{- /*
Zenith Helm Template Helpers
Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
*/ -}}

{{/*
Expand the name of the chart.
*/}}
{{- define "zenith.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "zenith.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "zenith.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "zenith.labels" -}}
helm.sh/chart: {{ include "zenith.chart" . }}
{{ include "zenith.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "zenith.selectorLabels" -}}
app.kubernetes.io/name: {{ include "zenith.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "zenith.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "zenith.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the PVC to use
*/}}
{{- define "zenith.pvcName" -}}
{{- if .Values.triton.modelRepository.pvc.existingClaim }}
{{- .Values.triton.modelRepository.pvc.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "zenith.fullname" .) }}
{{- end }}
{{- end }}
