brew install sonar-scanner
sonar-scanner \
  -Dsonar.organization=niccolocorsani \
  -Dsonar.projectKey=niccolocorsani_Tesi \
  -Dsonar.sources=. \
  -Dsonar.host.url=https://sonarcloud.io