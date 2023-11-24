FROM node:20-alpine as build
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_BACKEND_URL=${BACKEND_URL}
ARG VITE_FRONTEND_PORT=${FRONTEND_PORT}
RUN touch .env
RUN printenv > .env

RUN npm run build

FROM nginx:latest as proxy
COPY --from=build /usr/src/app/dist/ /usr/share/nginx/html
COPY --from=build /usr/src/app/nginx/nginx.conf /etc/nginx/conf.d/default.conf
CMD ["nginx", "-g", "daemon off;"]
EXPOSE 80
