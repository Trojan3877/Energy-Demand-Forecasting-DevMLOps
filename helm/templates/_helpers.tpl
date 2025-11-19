{{/*
==============================================
  Helm Template Helpers for Naming & Metadata
==============================================
*/}}

{{/*
Return the chart name.
*/}}
{{- define "energy-demand-forecast.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Return the fully qualified app name.
If release name contains the chart name, use it directly.
*/}}
{{- define "energy-demand-forecast.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "energy-demand-forecast.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels applied to all resources.
*/}}
{{- define "energy-demand-forecast.labels" -}}
app.kubernetes.io/name: {{ include "energy-demand-forecast.name" . }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels — must match in Deployment/Service.
*/}}
{{- define "energy-demand-forecast.selectorLabels" -}}
app: {{ include "energy-demand-forecast.name" . }}
{{- end -}}

{{/*
Standard metadata block for reuse.
*/}}
{{- define "energy-demand-forecast.metadata" -}}
labels:
  {{- include "energy-demand-forecast.labels" . | nindent 2 }}
{{- end -}}
